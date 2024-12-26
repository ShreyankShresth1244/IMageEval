import cv2
import numpy as np
from PIL import Image
from rembg import remove
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_resolution(image, min_resolution=(1024, 1024)):
    """
    Check if the image meets the minimum resolution requirement.
    """
    if len(image.shape) == 2:  # Grayscale image
        height, width = image.shape
    else:  # Color image
        height, width, _ = image.shape

    if height < min_resolution[1] or width < min_resolution[0]:
        return False, "Low resolution"
    return True, "Resolution OK"


def check_clarity(image, sharpness_threshold=100.0):
    """
    Check image clarity using Laplacian variance.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    if laplacian_var < sharpness_threshold:
        return False, "Blurry", laplacian_var
    return True, "Clarity OK", laplacian_var


def check_background(image_path, tolerance=10, max_unique_colors=10):
    """
    Check if the image background is plain (e.g., white).
    """
    try:
        img = Image.open(image_path)
        bg_removed = remove(img)
        bg_array = np.array(bg_removed.resize((100, 100)))  # Downscale for performance
        unique_colors = len(np.unique(bg_array.reshape(-1, bg_array.shape[2]), axis=0))

        if unique_colors > max_unique_colors:
            return False, "Complex background"
        return True, "Background OK"
    except Exception as e:
        logging.error(f"Error during background check: {e}")
        return False, "Error checking background"


def evaluate_image(image_path, min_resolution=(1024, 1024), sharpness_threshold=100.0):
    """
    Evaluate image based on resolution, clarity, and background.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded.")

        issues = []
        metadata = {}

        # Check resolution
        resolution_ok, resolution_message = check_resolution(image, min_resolution)
        if not resolution_ok:
            issues.append(resolution_message)
        metadata["resolution"] = image.shape[:2]

        # Check clarity
        clarity_ok, clarity_message, sharpness = check_clarity(image, sharpness_threshold)
        if not clarity_ok:
            issues.append(clarity_message)
        metadata["sharpness_score"] = sharpness

        # Check background
        background_ok, background_message = check_background(image_path)
        if not background_ok:
            issues.append(background_message)

        status = "Good" if not issues else "Needs Improvement"
        return {
            "status": status,
            "issues": issues,
            "metadata": metadata,
        }
    except Exception as e:
        logging.error(f"Error evaluating image: {e}")
        return {"status": "Invalid Image", "issues": [str(e)]}


# Test the function
if __name__ == "__main__":
    # Test with a sample image
    sample_image_path = "./data/original/sample_image.jpg"  # Change to your test image path
    result = evaluate_image(sample_image_path)
    logging.info(f"Evaluation Result: {result}")