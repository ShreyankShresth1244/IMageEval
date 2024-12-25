import os
import logging
from flask import Flask, request, jsonify
from app.config import STORAGE_PATHS
from app.enhancement import enhance_image
from app.utils import download_image  # Import the download_image function

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

@app.route("/enhance", methods=["POST"])
def enhance():
    """
    Endpoint to enhance an image using ESRGAN.

    Request Body:
        {
            "image_url": "https://example.com/image.jpg"
        }

    Response:
        {
            "enhanced_image_path": "./data/enhanced/image.jpg"
        }
    """
    try:
        # Parse request data
        data = request.json
        if not data or not data.get("image_url"):
            logging.error("Image URL is missing in the request.")
            return jsonify({"error": "Image URL is required"}), 400

        image_url = data["image_url"]

        # Validate URL format
        if not image_url.lower().endswith(("jpg", "jpeg", "png")):
            logging.error(f"Unsupported image format for URL: {image_url}")
            return jsonify({"error": "Unsupported image format. Only JPG and PNG are allowed."}), 400

        # Define paths for original and enhanced images
        image_path = os.path.join(STORAGE_PATHS["original"], os.path.basename(image_url))
        save_path = os.path.join(STORAGE_PATHS["enhanced"], os.path.basename(image_url))

        # Download the image
        try:
            download_image(image_url, image_path)
            logging.info(f"Image downloaded successfully: {image_path}")
        except Exception as e:
            logging.error(f"Failed to download image from URL: {image_url}. Error: {e}")
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 500

        # Enhance the image
        try:
            enhanced_path = enhance_image(image_path, save_path, "../models/esrgan/weights/")
            logging.info(f"Image enhanced successfully: {enhanced_path}")
        except Exception as e:
            logging.error(f"Failed to enhance image: {image_path}. Error: {e}")
            return jsonify({"error": f"Failed to enhance image: {str(e)}"}), 500

        return jsonify({"enhanced_image_path": enhanced_path}), 200

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0", port=5000, debug=True)
