import logging
from scripts.fetch_images import fetch_image_urls, save_images_locally
from scripts.batch_process import process_images_in_batches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./data/logs/test_pipeline.log"),
        logging.StreamHandler()
    ]
)

def test_pipeline():
    """
    Test the end-to-end pipeline: Fetch, process, and evaluate images.
    """
    try:
        # Fetch and save images locally
        logging.info("Fetching image URLs from database...")
        image_urls = fetch_image_urls()
        if not image_urls:
            logging.warning("No image URLs fetched from the database.")
            return

        logging.info(f"Fetched {len(image_urls)} image URL entries.")
        save_images_locally(image_urls)
        logging.info("Image download completed.")

        # Process images in batches
        logging.info("Processing images in batches...")
        process_images_in_batches()
        logging.info("Batch processing completed successfully.")

    except Exception as e:
        logging.error(f"Error during pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    test_pipeline()
