import logging
import os
import numpy as np
from PIL import Image
from rembg import remove
import cv2
import torch
from torchvision.transforms import functional as TF
from models.esrgan import ESRGANModel


def load_esrgan_model(model_path):
    """
    Load and initialize the ESRGAN model once.
    Args:
        model_path (str): Path to the ESRGAN model weights.
    Returns:
        ESRGANModel: Loaded ESRGAN model instance.
    """
    model = ESRGANModel(model_path)
    return model


def preprocess_small_image(image, target_size=(256, 256)):
    """
    Preprocess small images by upscaling them to a minimum size using bicubic interpolation.
    Args:
        image (PIL.Image.Image): Input image as a PIL Image.
        target_size (tuple): Target size for upscaling (width, height).
    Returns:
        PIL.Image.Image: Upscaled image.
    """
    return image.resize(target_size, Image.BICUBIC)



def sharpen_image(image):
    """
    Apply a sharpening filter to the image.
    Args:
        image (ndarray): Input image as a NumPy array.
    Returns:
        ndarray: Sharpened image as a NumPy array.
    """
    if image.shape[0] < 256 or image.shape[1] < 256:
        return image  # Skip sharpening for very small images
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def upscale_image_with_esrgan(image, model):
    """
    Upscale the image using a preloaded ESRGAN model.
    Args:
        image (PIL.Image.Image): Input image as a PIL Image.
        model (ESRGANModel): Preloaded ESRGAN model instance.
    Returns:
        PIL.Image.Image: Enhanced image after upscaling.
    """
    if model is None:
        raise ValueError("ESRGAN model is not loaded.")

    image_tensor = TF.to_tensor(image).unsqueeze(0)  # Convert image to PyTorch tensor
    with torch.no_grad():
        enhanced_tensor = model(image_tensor)  # Pass through the ESRGAN model
    enhanced_image = TF.to_pil_image(enhanced_tensor.squeeze(0))
    return enhanced_image


def replace_background(image):
    """
    Replace the background of the image with a plain white background.
    Args:
        image (PIL.Image.Image): Input image as a PIL Image.
    Returns:
        PIL.Image.Image: Image with white background (RGB mode).
    """
    if image.size[0] < 256 or image.size[1] < 256:
        return image  # Skip background removal for very small images

    # Remove the original background
    bg_removed = remove(image)

    # Ensure the result is in 'RGBA' mode to handle transparency
    if bg_removed.mode != "RGBA":
        bg_removed = bg_removed.convert("RGBA")

    # Create a new white background (RGB mode)
    white_bg = Image.new("RGBA", bg_removed.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(white_bg, bg_removed)

    # Convert back to 'RGB' mode (discard alpha channel)
    return combined.convert("RGB")



def enhance_image(image_path, save_path, esrgan_model):
    """
    Enhance the image by sharpening, upscaling, and replacing the background.
    Args:
        image_path (str): Path to the original image.
        save_path (str): Path to save the enhanced image.
        esrgan_model (ESRGANModel): Preloaded ESRGAN model.
    Returns:
        str: Path to the saved enhanced image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found locally: {image_path}")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read the image. Invalid image path.")

    try:
        # Sharpen the image
        sharpened = sharpen_image(image)

        # Convert OpenCV image (NumPy array) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

        # Preprocess small images
        if pil_image.size[0] < 256 or pil_image.size[1] < 256:
            pil_image = preprocess_small_image(pil_image)

        # Upscale the image using ESRGAN with mixed precision
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                upscaled = upscale_image_with_esrgan(pil_image, esrgan_model)

        # Replace the background with plain white
        enhanced_image = replace_background(upscaled)

        # Save the enhanced image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        enhanced_image.save(save_path)

    except Exception as e:
        logging.error(f"Error during enhancement: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

    return save_path
