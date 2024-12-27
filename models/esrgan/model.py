import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image
import os
import logging
from multiprocessing import Pool

from models.esrgan.ESRGAN.RRDBNet_arch import RRDBNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ESRGANModel(nn.Module):
    """
    ESRGAN Model for Super-Resolution.
    """

    def __init__(self, weights_path, in_nc=3, out_nc=3, nf=64, nb=23):
        super(ESRGANModel, self).__init__()
        self.model = self.load_pretrained_model(weights_path, in_nc, out_nc, nf, nb)

    def load_pretrained_model(self, weights_path, in_nc, out_nc, nf, nb):
        logging.info("Loading ESRGAN model...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        try:
            model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)
            checkpoint = torch.load(weights_path, map_location=torch.device("cuda"), weights_only=True)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model = model.cuda()
            model.eval()

            end_event.record()
            torch.cuda.synchronize()
            logging.info(f"Model loaded and moved to GPU in {start_event.elapsed_time(end_event):.2f} ms")
            return model
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            raise ValueError(f"Error loading model from path: {weights_path}")

    def forward(self, x):
        return self.model(x.cuda())  # Move the input tensor to GPU


def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_tensor = TF.to_tensor(image).unsqueeze(0).cuda()  # Move to GPU
    return image_tensor



def postprocess_image(image_tensor, save_path):
    enhanced_image = TF.to_pil_image(image_tensor.cpu().squeeze(0).clamp(0, 1))  # Move to CPU
    enhanced_image.save(save_path)



def test_esrgan_model(image_path, model_path, save_path, model=None):
    """
    Test the ESRGAN model with a sample image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if model is None and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    if model is None:
        model = ESRGANModel(model_path)

    input_tensor = preprocess_image(image_path)

    try:
        with torch.no_grad():
            output_tensor = model(input_tensor)

        postprocess_image(output_tensor, save_path)
        logging.info(f"Enhanced image saved at {save_path}")
    except Exception as e:
        logging.error(f"Failed to process image {image_path}: {e}")


def process_image(args):
    """
    Helper function for multiprocessing.
    """
    image_path, save_path, model_path, model = args
    try:
        test_esrgan_model(image_path, model_path, save_path, model)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    model_path = "./models/esrgan/weights/RRDB_ESRGAN_x4.pth"
    model = ESRGANModel(model_path)

    # Define image paths
    test_images = [
        ("./data/original/test_image1.jpg", "./data/enhanced/enhanced_test_image1.jpg"),
        ("./data/original/test_image2.jpg", "./data/enhanced/enhanced_test_image2.jpg"),
    ]

    # Add model to each argument tuple for multiprocessing
    args_list = [(img[0], img[1], model_path, model) for img in test_images]

    # Use multiprocessing for batch processing
    with Pool(processes=4) as pool:
        pool.map(process_image, args_list)