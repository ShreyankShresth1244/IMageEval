import os
import pytest
from unittest.mock import patch
from app.enhancement import enhance_image

@pytest.fixture
def test_image_paths():
    """
    Fixture to provide test input and output paths.
    """
    input_path = "./data/original/test_low_res.jpg"
    output_path = "./data/enhanced/test_enhanced.jpg"
    yield input_path, output_path
    # Clean up after the test
    if os.path.exists(output_path):
        os.remove(output_path)

@patch("app.enhancement.upscale_image_with_esrgan")
def test_enhance_image(mock_upscale, test_image_paths):
    """
    Test the enhance_image function with a valid input.
    """
    # Mock the upscale function to return a placeholder
    mock_upscale.return_value = "./data/enhanced/test_enhanced.jpg"

    input_path, output_path = test_image_paths

    # Ensure input file exists for testing
    assert os.path.exists(input_path), f"Test input file not found: {input_path}"

    # Run the enhance_image function
    result_path = enhance_image(input_path, output_path)

    # Assertions
    assert result_path == output_path, "Output path does not match expected path."
    assert os.path.exists(output_path), "Enhanced image file does not exist."
    mock_upscale.assert_called_once_with(input_path, output_path)

def test_enhance_image_invalid_input():
    """
    Test the enhance_image function with an invalid input path.
    """
    input_path = "./data/original/non_existent.jpg"
    output_path = "./data/enhanced/test_enhanced_invalid.jpg"

    with pytest.raises(ValueError, match="Invalid image path."):
        enhance_image(input_path, output_path)
