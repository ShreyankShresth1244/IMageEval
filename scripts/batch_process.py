import os
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
from app.evaluation import evaluate_image
from app.enhancement import enhance_image
from app.config import STORAGE_PATHS, BATCH_PROCESSING

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(STORAGE_PATHS["logs"], "batch_processing.log")),
        logging.StreamHandler()
    ]
)

def process_image(image_file):
    """
    Process a single image: Evaluate and enhance if needed.

    Args:
        image_file (str): File name of the image to process.
    """
    image_path = os.path.join(STORAGE_PATHS["original"], image_file)
    enhanced_path = os.path.join(STORAGE_PATHS["enhanced"], image_file)

    # Check if the image exists locally before processing
    if not os.path.exists(image_path):
        logging.error(f"Image not found locally: {image_path}")
        return

    try:
        # Evaluate the image
        logging.info(f"Evaluating image: {image_file}")
        evaluation_result = evaluate_image(image_path)
        logging.info(f"Evaluation for {image_file}: {evaluation_result}")

        # Enhance if the image needs improvement
        if evaluation_result["status"] == "Needs Improvement":
            logging.info(f"Enhancing image: {image_file}")
            enhance_image(image_path, enhanced_path)
            logging.info(f"Enhanced {image_file} saved at {enhanced_path}")
        else:
            logging.info(f"{image_file} does not need enhancement.")
    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")

# batch_process.py

def get_dynamic_batch_size():
    """
    Returns the dynamic batch size for processing.
    """
    return 5  # or some logic to determine the batch size dynamically


def process_images_in_batches_parallel():
    """
    Process images in batches using parallel workers.
    """
    # Get all valid image files in the original directory
    original_images = [
        f for f in os.listdir(STORAGE_PATHS["original"])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not original_images:
        logging.warning("No images found in the original directory.")
        return

    batch_size = BATCH_PROCESSING["batch_size"]

    for i in range(0, len(original_images), batch_size):
        batch = original_images[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1}: {len(batch)} images")

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=BATCH_PROCESSING["parallel_workers"]) as executor:
            executor.map(process_image, batch)

        logging.info(f"Batch {i // batch_size + 1} completed.")

if __name__ == "__main__":
    try:
        logging.info("Starting batch processing.")
        process_images_in_batches_parallel()
        logging.info("All batches processed successfully.")
    except Exception as e:
        logging.error(f"Critical error during batch processing: {e}")