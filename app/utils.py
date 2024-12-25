import os
import requests
import logging
from urllib.parse import urlparse, quote
import time


def create_directories(paths):
    """
    Create directories if they do not exist.

    Args:
        paths (list): List of directory paths to create.
    """
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Directory ensured: {path}")
        except Exception as e:
            logging.error(f"Failed to ensure directory: {path}. Error: {e}")


def is_valid_url(url):
    """
    Validate if the given string is a valid URL.

    Args:
        url (str): URL string to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def encode_url(url):
    """
    Encode the URL to handle special characters.

    Args:
        url (str): The URL to be encoded.

    Returns:
        str: The encoded URL.
    """
    # Handle URL encoding while ensuring ':' and '/' are safe
    encoded_url = quote(url, safe=':/')
    return encoded_url





def download_image(url, save_path):
    """
    Download an image from a URL and save it locally.

    Args:
        url (str): The URL of the image to download.
        save_path (str): The local file path to save the downloaded image.

    Raises:
        ValueError: If the URL is invalid or the download fails.
    """
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    # Encode the URL to handle special characters
    encoded_url = encode_url(url)
    logging.info(f"Encoded URL: {encoded_url}")

    try:
        response = requests.get(encoded_url, stream=True, timeout=10)
        # Check for HTTP errors
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        logging.info(f"Image downloaded successfully: {save_path}")

        # Add a delay to handle rate limiting
        time.sleep(1)  # Adjust the delay as needed (e.g., 1 second)

    except requests.exceptions.HTTPError as e:
        logging.error(f"Failed to download image from {encoded_url}. HTTP error: {e}")
        raise ValueError(f"Failed to download image from {encoded_url}. HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Network or URL request error when downloading from {encoded_url}: {e}")
        raise ValueError(f"Network or URL request error when downloading from {encoded_url}: {e}")
    except Exception as e:
        logging.error(f"Failed to download image from {encoded_url}. Error: {e}")
        raise ValueError(f"Failed to download image from {encoded_url}. Error: {e}")


def configure_logging(log_file):
    """
    Configure logging settings.

    Args:
        log_file (str): Path to the log file.
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info(f"Logging configured: {log_file}")
    except Exception as e:
        logging.error(f"Failed to configure logging. Error: {e}")
        print(f"Failed to configure logging. Error: {e}")


def log_message(level, message):
    """
    Log a message at a specified level.

    Args:
        level (str): Logging level (e.g., 'info', 'error').
        message (str): Message to log.
    """
    level = level.lower()
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "debug":
        logging.debug(message)
    else:
        logging.error(f"Invalid logging level: {level}")
    logging.info(f"Logged [{level.upper()}]: {message}")


def validate_file_exists(file_path):
    """
    Validate if a file exists at the given path.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if os.path.exists(file_path):
        logging.info(f"File exists: {file_path}")
        return True
    logging.warning(f"File does not exist: {file_path}")
    return False
