import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image
import os
import logging

from models.esrgan.ESRGAN.RRDBNet_arch import RRDBNet


# Assuming RRDBNet is defined in a separate file or repository  # Adjust the import based on where RRDBNet is defined

class ESRGANModel(nn.Module):
    """
    ESRGAN Model for Super-Resolution.
    """

    def __init__(self, weights_path, in_nc=3, out_nc=3, nf=64, nb=23):
        super(ESRGANModel, self).__init__()
        self.model = self.load_pretrained_model(weights_path, in_nc, out_nc, nf, nb)

    def load_pretrained_model(self, weights_path, in_nc, out_nc, nf, nb):
        """
        Load the pre-trained ESRGAN model from the given path.
        """
        logging.info("Loading ESRGAN model...")
        try:
            # Define the ESRGAN architecture by importing RRDBNet and passing the necessary arguments
            model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)  # Pass the required parameters

            # Load the weights
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)  # If no 'state_dict', load directly

            model.eval()  # Set the model to evaluation mode
            logging.info("Model loaded successfully.")
            return model

        except KeyError as e:
            logging.error(f"KeyError while loading model: {e}")
            raise ValueError(f"Error loading model from path: {weights_path}")
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            raise ValueError(f"Error loading model from path: {weights_path}")

    def forward(self, x):
        """
        Forward pass for the ESRGAN model.
        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Super-resolved image tensor.
        """
        return self.model(x)


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an image for the ESRGAN model.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image to (width, height).
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    return image_tensor


def postprocess_image(image_tensor, save_path):
    """
    Postprocess the model output and save it as an image.
    Args:
        image_tensor (torch.Tensor): Super-resolved image tensor.
        save_path (str): Path to save the output image.
    """
    enhanced_image = TF.to_pil_image(image_tensor.squeeze(0).clamp(0, 1))
    enhanced_image.save(save_path)


def test_esrgan_model(image_path, model_path, save_path):
    """
    Test the ESRGAN model with a sample image.
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the ESRGAN model weights.
        save_path (str): Path to save the enhanced image.
    """
    # Check if image and model path exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Initialize the ESRGAN model
    model = ESRGANModel(model_path)

    # Preprocess the image
    input_tensor = preprocess_image(image_path)

    # Perform super-resolution without gradient tracking
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess and save the enhanced image
    postprocess_image(output_tensor, save_path)
    print(f"Enhanced image saved at {save_path}")


if __name__ == "__main__":
    # Test the ESRGAN model with a sample image
    image_path = "./data/original/test_image.jpg"  # Path to input image
    model_path = "./models/esrgan/weights/RRDB_ESRGAN_x4.pth"  # Path to model weights
    save_path = "./data/enhanced/enhanced_test_image.jpg"  # Path to save enhanced image

    test_esrgan_model(image_path, model_path, save_path)