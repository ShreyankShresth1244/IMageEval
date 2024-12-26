import os
import logging
from concurrent.futures import ThreadPoolExecutor
from app.evaluation import evaluate_image
from app.enhancement import enhance_image, load_esrgan_model
from app.config import STORAGE_PATHS, BATCH_PROCESSING
import psutil

logging.info(f"Memory usage before enhancement: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")


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
    Args:
        image_list (list): List of image file names.
        batch_size (int): Number of images in each batch.
    Returns:
        list: A list of batches, where each batch is a list of image file names.
    """
    return [image_list[i:i + batch_size] for i in range(0, len(image_list), batch_size)]



def process_image(image_file, esrgan_model):
    """
    Process a single image: Evaluate and enhance if needed.
    Args:
        image_file (str): File name of the image to process.
        esrgan_model: Preloaded ESRGAN model for upscaling.
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

        if evaluation_result["status"] == "Needs Improvement":
            logging.info(f"Enhancing image: {image_file}")
            enhance_image(image_path, enhanced_path, esrgan_model)
            logging.info(f"Enhanced {image_file} saved at {enhanced_path}")
        else:
            logging.info(f"{image_file} does not need enhancement.")
    except Exception as e:
        logging.error(f"Error processing {image_file}: {e}")


def process_images_in_batches_parallel():
    """
    Process images in batches using parallel workers.
    """
    original_images = [
        f for f in os.listdir(STORAGE_PATHS["original"])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not original_images:
        logging.warning("No images found in the original directory.")
        return

    batch_size = BATCH_PROCESSING["batch_size"]
    parallel_workers = min(2, BATCH_PROCESSING["parallel_workers"])  # Cap workers to 2

    # Load the ESRGAN model once before batch processing
    logging.info("Loading ESRGAN model...")
    esrgan_model = load_esrgan_model("./models/esrgan/weights/RRDB_ESRGAN_x4.pth")
    logging.info("ESRGAN model loaded successfully.")

    # Get batches using the get_batches function
    batches = get_batches(original_images, batch_size)

    for batch_number, batch in enumerate(batches, start=1):
        logging.info(f"Processing batch {batch_number}: {len(batch)} images")

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            executor.map(lambda img: process_image(img, esrgan_model), batch)

        logging.info(f"Batch {batch_number} completed.")


if __name__ == "__main__":
    try:
        logging.info("Starting batch processing.")
        process_images_in_batches_parallel()
        logging.info("All batches processed successfully.")
    except Exception as e:
        logging.error(f"Critical error during batch processing: {e}")