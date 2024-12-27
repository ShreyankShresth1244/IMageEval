import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import gc
from numba import NumbaWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=NumbaWarning)

import torch

from app.evaluation import evaluate_image
from app.enhancement import enhance_image, load_esrgan_model
from app.config import STORAGE_PATHS, BATCH_PROCESSING
from app.database import get_db_connection
import psutil
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(STORAGE_PATHS["logs"], "batch_processing.log")),
        logging.StreamHandler(),
    ],
)

def get_batches(image_list, batch_size):
    """
    Split the list of images into smaller batches.
    """
    return [image_list[i:i + batch_size] for i in range(0, len(image_list), batch_size)]

def get_original_url(image_file, conn):
    """
    Get the original URL of the image from the `products` table if available.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT image_url1 FROM products WHERE image_url1 LIKE %s
                """,
                (f"%{image_file}%",)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
    except Exception as e:
        logging.error(f"Error fetching original URL for {image_file}: {e}")
    return os.path.abspath(os.path.join(STORAGE_PATHS["original"], image_file))

def save_to_database(conn, original_url, enhanced_path, evaluation_status, issues_detected):
    """
    Save the image details to the database.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO enhanced_images (original_image_url, enhanced_image_url, evaluation_status, issues_detected)
                VALUES (%s, %s, %s, %s::json)
                """,
                (original_url, enhanced_path, evaluation_status, json.dumps(issues_detected)),
            )
            conn.commit()
            logging.info(f"Saved enhanced image details to database: {enhanced_path}")
    except Exception as e:
        logging.error(f"Error saving to database: {e}")

def process_image(image_file, esrgan_model, conn):
    """
    Process a single image: Evaluate and enhance if needed, and save details to the database.
    """
    image_path = os.path.join(STORAGE_PATHS["original"], image_file)
    enhanced_path = os.path.join(STORAGE_PATHS["enhanced"], image_file)

    if not os.path.exists(image_path):
        logging.error(f"Image not found locally: {image_path}")
        return

    try:
        logging.info(f"Evaluating image: {image_file}")
        evaluation_result = evaluate_image(image_path)
        logging.info(f"Evaluation result for {image_file}: {evaluation_result}")

        original_url = get_original_url(image_file, conn)
        if evaluation_result["status"] == "Needs Improvement":
            logging.info(f"Enhancing image: {image_file}")
            enhance_image(image_path, enhanced_path, esrgan_model)
            logging.info(f"Enhanced {image_file} saved at {enhanced_path}")
        else:
            logging.info(f"{image_file} does not need enhancement.")
            enhanced_path = None

        save_to_database(
            conn=conn,
            original_url=original_url,
            enhanced_path=enhanced_path,
            evaluation_status=evaluation_result["status"],
            issues_detected=evaluation_result.get("issues", {})
        )
    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")
    finally:
        # Clear memory after processing each image
        gc.collect()
        torch.cuda.empty_cache()

def measure_performance(func):
    """
    Decorator to measure performance of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Performance: {func.__name__} took {elapsed_time:.2f} seconds")
        return result
    return wrapper

def process_images_in_batches_parallel():
    original_images = [
        f for f in os.listdir(STORAGE_PATHS["original"])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not original_images:
        logging.warning("No images found in the original directory.")
        return

    # Get batch size and parallel workers from config
    batch_size = BATCH_PROCESSING.get("batch_size", 4)  # Default to 4 if not specified in config
    parallel_workers = min(2, BATCH_PROCESSING.get("parallel_workers", 2))  # Default to 2 workers if not specified

    logging.info(f"Batch size: {batch_size}, Parallel workers: {parallel_workers}")

    logging.info("Loading ESRGAN model...")
    # Clear memory before loading the model
    gc.collect()
    torch.cuda.empty_cache()
    esrgan_model = load_esrgan_model("./models/esrgan/weights/RRDB_ESRGAN_x4.pth").cuda()
    esrgan_model.half()
    logging.info("ESRGAN model loaded successfully.")

    conn = get_db_connection()

    try:
        batches = get_batches(original_images, batch_size)
        total_start_time = time.time()  # Start total processing time tracking

        for batch_number, batch in enumerate(batches, start=1):
            logging.info(f"Processing batch {batch_number}: {len(batch)} images")

            # Start timing for batch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                executor.map(lambda img: process_image(img, esrgan_model, conn), batch)

            end_event.record()
            torch.cuda.synchronize()
            logging.info(f"Batch {batch_number} completed in {start_event.elapsed_time(end_event):.2f} ms.")

            # Clear memory after processing each batch
            gc.collect()
            torch.cuda.empty_cache()

        total_end_time = time.time()  # End total processing time tracking
        total_elapsed_time = total_end_time - total_start_time
        logging.info(f"Total time taken to process all batches: {total_elapsed_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"Critical error during batch processing: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        logging.info("Starting batch processing.")
        process_images_in_batches_parallel()
        logging.info("All batches processed successfully.")
    except Exception as e:
        logging.error(f"Critical error during batch processing: {e}")
