import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from app.enhancement import enhance_image, sharpen_image, replace_background, upscale_image_with_esrgan

# Define sample paths
TEST_IMAGE_PATH = "./tests/data/test_image.jpg"
SAVE_IMAGE_PATH = "./tests/data/enhanced_test_image.jpg"


@pytest.fixture
def mock_image():
    # Create a mock image
    width, height = 100, 100
    return Image.new("RGB", (width, height), (0, 0, 0))


def test_sharpen_image():
    # Test sharpen_image
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image
    sharpened = sharpen_image(img_array)
    assert sharpened.shape == img_array.shape
    assert sharpened.dtype == img_array.dtype


@patch("app.enhancement.remove")
def test_replace_background(mock_remove):
    # Mock the background removal
    mock_bg_removed = Image.new("RGB", (100, 100), (0, 0, 0))  # Mock black image in RGB
    mock_remove.return_value = mock_bg_removed

    # Create a mock input image
    mock_image = Image.new("RGB", (100, 100), (0, 0, 0))  # Black image

    # Call the function and verify
    result = replace_background(mock_image)

    assert isinstance(result, Image.Image)
    assert result.size == mock_bg_removed.size
    assert result.mode == "RGB"  # Ensure the mode is RGB

@patch("app.enhancement.ESRGANModel")
@patch("torch.no_grad")
@patch("app.enhancement.TF.to_tensor")
@patch("app.enhancement.TF.to_pil_image")
def test_upscale_image_with_esrgan(mock_to_pil_image, mock_to_tensor, mock_no_grad, mock_esrgan_model):
    # Mock ESRGAN processing
    mock_model = MagicMock()
    mock_esrgan_model.return_value = mock_model

    mock_tensor = MagicMock()
    mock_to_tensor.return_value = mock_tensor

    mock_to_pil_image.return_value = Image.new("RGB", (200, 200), (0, 0, 0))  # Upscaled image

    result = upscale_image_with_esrgan(Image.new("RGB", (100, 100), (0, 0, 0)), mock_model)
    assert isinstance(result, Image.Image)
    assert result.size == (200, 200)  # Ensure it's upscaled





@patch("os.path.exists")
@patch("os.makedirs")
@patch("cv2.imread")
@patch("app.enhancement.replace_background")
@patch("app.enhancement.upscale_image_with_esrgan")
@patch("app.enhancement.sharpen_image")
def test_enhance_image(
    mock_sharpen,
    mock_upscale,
    mock_replace_bg,
    mock_cv2_read,
    mock_makedirs,
    mock_path_exists,
):
    # Mock dependencies
    mock_path_exists.return_value = True
    mock_image = Image.new("RGB", (100, 100), (0, 0, 0))
    mock_sharpen.return_value = np.zeros((100, 100, 3), dtype=np.uint8)  # Sharpened image array
    mock_upscale.return_value = mock_image
    mock_replace_bg.return_value = mock_image
    mock_cv2_read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock the save method
    mock_image.save = MagicMock()

    # Mock ESRGAN model
    mock_esrgan_model = MagicMock()

    # Call the function
    from app.enhancement import enhance_image

    result = enhance_image(TEST_IMAGE_PATH, SAVE_IMAGE_PATH, mock_esrgan_model)

    # Assertions
    assert result is not None