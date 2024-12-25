import cv2
from PIL import Image
from rembg import remove
import numpy as np
import torch
from torchvision.transforms import functional as TF
from models.esrgan import ESRGANModel  # Import from esrgan package
import os

model_path = "./models/esrgan/weights/RRDB_ESRGAN_x4.pth"

def sharpen_image(image):
    """
    Apply sharpening filter to the image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def replace_background(image_path):
    """
    Replace the image background with plain white.
    """
    img = Image.open(image_path)
    bg_removed = remove(img)
    white_bg = Image.new("RGB", bg_removed.size, (255, 255, 255))
    white_bg.paste(bg_removed, (0, 0), bg_removed)
    return white_bg


def upscale_image_with_esrgan(image, model_path="./models/esrgan/weights/RRDB_ESRGAN_x4.pth"):
    """
    Use the ESRGAN model to upscale the image.
    """
    model = ESRGANModel(model_path)
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        enhanced_tensor = model(image_tensor)
    enhanced_image = TF.to_pil_image(enhanced_tensor.squeeze(0))
    return enhanced_image


def enhance_image(image_path, save_path, model_path="./models/esrgan/weights/RRDB_ESRGAN_x4.pth"):
    """
    Enhance the image by sharpening, upscaling using ESRGAN, and replacing the background.
    """
    # Check if image exists locally
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found locally: {image_path}")

    # Read the image using OpenCV (only if it exists locally)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path.")

    # Sharpen the image
    sharpened = sharpen_image(image)

    # Upscale the image using ESRGAN
    upscaled = upscale_image_with_esrgan(Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)), model_path)

    # Replace the background with a white background
    white_bg = replace_background(image_path)

    # Save the enhanced image
    enhanced_path = save_path
    if not os.path.exists(os.path.dirname(enhanced_path)):
        os.makedirs(os.path.dirname(enhanced_path))
    white_bg.save(enhanced_path)

    return enhanced_path
