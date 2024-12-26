import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
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

def get_batches(image_list, batch_size=2):
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

@measure_performance
def process_images_in_batches_parallel():
    """
    Process images in batches using parallel workers, measuring performance.
    """
    original_images = [
        f for f in os.listdir(STORAGE_PATHS["original"])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not original_images:
        logging.warning("No images found in the original directory.")
        return

    batch_size = BATCH_PROCESSING["batch_size"]
    parallel_workers = min(2, BATCH_PROCESSING["parallel_workers"])

    # Load the ESRGAN model once before batch processing
    logging.info("Loading ESRGAN model...")
    esrgan_model = load_esrgan_model("./models/esrgan/weights/RRDB_ESRGAN_x4.pth")
    logging.info("ESRGAN model loaded successfully.")

    # Establish a single database connection
    conn = get_db_connection()

    try:
        batches = get_batches(original_images, batch_size)
        for batch_number, batch in enumerate(batches, start=1):
            logging.info(f"Processing batch {batch_number}: {len(batch)} images")
            batch_start_time = time.time()

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                executor.map(lambda img: process_image(img, esrgan_model, conn), batch)

            batch_end_time = time.time()
            logging.info(f"Batch {batch_number} completed in {batch_end_time - batch_start_time:.2f} seconds.")
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
