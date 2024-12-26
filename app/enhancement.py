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


def sharpen_image(image):
    """
    Apply a sharpening filter to the image.
    Args:
        image (ndarray): Input image as a NumPy array.
    Returns:
        ndarray: Sharpened image as a NumPy array.
    """
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
    Replace the background of an image with a plain white background.
    Args:
        image (PIL.Image.Image): Input image as a PIL Image.
    Returns:
        PIL.Image.Image: Image with the background replaced.
    """
    bg_removed = remove(image)
    white_bg = Image.new("RGB", bg_removed.size, (255, 255, 255))
    white_bg.paste(bg_removed, (0, 0), bg_removed)
    return white_bg


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

    # Sharpen the image
    sharpened = sharpen_image(image)

    # Convert OpenCV image (NumPy array) to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

    # Upscale the image using ESRGAN
    upscaled = upscale_image_with_esrgan(pil_image, esrgan_model)

    # Replace the background with plain white
    enhanced_image = replace_background(upscaled)

    # Save the enhanced image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    enhanced_image.save(save_path)

    return save_path
