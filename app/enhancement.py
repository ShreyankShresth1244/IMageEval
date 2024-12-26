import os
import cv2
from PIL import Image
import numpy as np
import torch
from rembg import remove
from torchvision.transforms import functional as TF
from models.esrgan import ESRGANModel


def load_esrgan_model(model_path="./models/esrgan/weights/RRDB_ESRGAN_x4.pth"):
    """
    Load the ESRGAN model.
    """
    return ESRGANModel(model_path)


def sharpen_image(image):
    """
    Apply sharpening filter to the image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def upscale_image_with_esrgan(image, model):
    """
    Use the ESRGAN model to upscale the image.
    Args:
        image (PIL.Image): The input image to enhance.
        model: Pre-loaded ESRGAN model.
    Returns:
        PIL.Image: The upscaled image.
    """
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        enhanced_tensor = model(image_tensor)
    enhanced_image = TF.to_pil_image(enhanced_tensor.squeeze(0))
    return enhanced_image


def enhance_image(image_path, save_path, esrgan_model):
    """
    Enhance an image using sharpening, upscaling, and background replacement.
    Args:
        image_path (str): Path to the input image.
        save_path (str): Path to save the enhanced image.
        esrgan_model: Pre-loaded ESRGAN model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found locally: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path.")

    sharpened = sharpen_image(image)
    upscaled = upscale_image_with_esrgan(
        Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)),
        esrgan_model,
    )

    white_bg = replace_background(upscaled)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    white_bg.save(save_path)


def replace_background(image):
    """
    Replace the background with white for a given image.
    Args:
        image (PIL.Image): The input image with possibly a non-uniform background.
    Returns:
        PIL.Image: The processed image with a white background.
    """
    bg_removed = remove(image)
    white_bg = Image.new("RGB", bg_removed.size, (255, 255, 255))
    white_bg.paste(bg_removed, (0, 0), bg_removed)
    return white_bg
