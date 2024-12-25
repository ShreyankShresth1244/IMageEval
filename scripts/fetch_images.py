import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from app.config import DATABASE_CONFIG, STORAGE_PATHS
from app.utils import download_image, create_directories
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_valid_url(url):
    """
    Validate if the given string is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def fetch_image_urls():
    """
    Fetch image URLs from the products table in the database.
    """
    query = "SELECT id, image_url1, image_url2, image_url3, image_url4, image_url5 FROM products;"
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG, cursor_factory=RealDictCursor)
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    except Exception as e:
        logging.error(f"Error fetching image URLs from database: {e}")
        return []
    finally:
        if conn:
            conn.close()

def save_images_locally(image_urls):
    """
    Download images from URLs and save them locally.
    """
    create_directories([STORAGE_PATHS["original"]])
    for row in image_urls:
        for field in [f"image_url{i}" for i in range(1, 6)]:
            url = row[field]
            if url:
                # Validate URL
                if not is_valid_url(url):
                    logging.warning(f"Invalid URL skipped: {url}")
                    continue

                # Construct local file path
                image_path = os.path.join(STORAGE_PATHS["original"], f"{row['id']}_{os.path.basename(url)}")

                try:
                    # Download the image
                    download_image(url, image_path)
                    logging.info(f"Downloaded {url} to {image_path}")
                except Exception as e:
                    logging.error(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    try:
        # Fetch URLs from the database
        image_urls = fetch_image_urls()
        if not image_urls:
            logging.warning("No image URLs fetched from the database.")
        else:
            # Save images locally
            save_images_locally(image_urls)
            logging.info("All images downloaded successfully.")
    except Exception as e:
        logging.error(f"Critical error: {e}")
