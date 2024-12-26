import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.evaluation import check_resolution, check_clarity, check_background, evaluate_image
from PIL import Image
import cv2


@pytest.fixture
def mock_image():
    # Create a dummy image with some structure to avoid being "blurry"
    height, width = 1024, 1024
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(image, (100, 100), (900, 900), (255, 255, 255), 10)  # Add a diagonal white line
    return image


@pytest.fixture
def low_res_image():
    # Create a low-resolution RGB image
    height, width = 500, 500
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.fixture
def blurry_image():
    # Create a grayscale blurry image
    height, width = 1024, 1024
    blurry = cv2.GaussianBlur(np.zeros((height, width), dtype=np.uint8), (15, 15), 0)
    return cv2.cvtColor(blurry, cv2.COLOR_GRAY2BGR)


def test_check_resolution(mock_image, low_res_image):
    # Test valid resolution
    assert check_resolution(mock_image) == (True, "Resolution OK")

    # Test low resolution
    assert check_resolution(low_res_image) == (False, "Low resolution")


def test_check_clarity(mock_image, blurry_image):
    # Test clarity with a non-blurry image (sharpness threshold = 10 for this case)
    clear, clarity_message, sharpness_score = check_clarity(mock_image, sharpness_threshold=10)
    assert clear is True
    assert clarity_message == "Clarity OK"
    assert sharpness_score >= 10

    # Test blurry image
    clear, clarity_message, sharpness_score = check_clarity(blurry_image, sharpness_threshold=10)
    assert clear is False
    assert clarity_message == "Blurry"
    assert sharpness_score < 10


@patch("app.evaluation.Image.open")
@patch("app.enhancement.remove")
def test_check_background(mock_remove, mock_open):
    # Create a mock input image (all black)
    mock_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mock_open.return_value = Image.fromarray(mock_image)

    # Mock the background removal result as a plain white RGB image
    mock_removed_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    mock_remove.return_value = mock_removed_image

    # Call the function to test
    result, message = check_background("mock_path")

    # Debugging outputs
    print(f"[TEST DEBUG] Result: {result}, Message: {message}")
    unique_colors = len(
        np.unique(
            np.array(mock_removed_image.resize((100, 100))).reshape(-1, 3), axis=0
        )
    )
    print(f"[TEST DEBUG] Unique colors in mocked background: {unique_colors}")

    # Assertions
    assert result is True, f"Expected True but got {result}. Message: {message}"
    assert message == "Background OK"


@patch("cv2.imread")
@patch("app.evaluation.check_resolution")
@patch("app.evaluation.check_clarity")
@patch("app.evaluation.check_background")
def test_evaluate_image(mock_check_bg, mock_check_clarity, mock_check_res, mock_cv2_read, mock_image):
    # Mock all component functions and setup expected behaviors
    mock_cv2_read.return_value = mock_image
    mock_check_res.return_value = (True, "Resolution OK")
    mock_check_clarity.return_value = (True, "Clarity OK", 150.0)
    mock_check_bg.return_value = (True, "Background OK")

    result = evaluate_image("mock_path")

    # Validate the function's output
    assert result["status"] == "Good"
    assert len(result["issues"]) == 0
    assert result["metadata"]["resolution"] == (1024, 1024)
    assert result["metadata"]["sharpness_score"] == 150.0

    # Mock issues in components
    mock_check_res.return_value = (False, "Low resolution")
    mock_check_clarity.return_value = (False, "Blurry", 50.0)
    mock_check_bg.return_value = (False, "Complex background")
    result = evaluate_image("mock_path")
    assert result["status"] == "Needs Improvement"
    assert "Low resolution" in result["issues"]
    assert "Blurry" in result["issues"]
    assert "Complex background" in result["issues"]


@patch("cv2.imread")
def test_evaluate_image_invalid_path(mock_cv2_read):
    # Mock invalid image path scenario
    mock_cv2_read.return_value = None
    result = evaluate_image("invalid_path")
    assert result["status"] == "Invalid Image"
    assert "Image could not be loaded." in result["issues"]
