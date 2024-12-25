import pytest
from unittest.mock import patch, MagicMock
from batch_process import process_images_in_batches_parallel, process_image
from app.evaluation import evaluate_image, check_resolution, check_clarity, check_background
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@pytest.fixture
def mock_image_paths():
    """
    Fixture for mock test image paths.
    """
    return {
        "good_image": "./tests/data/mock_good_image.jpg",
        "blurry_image": "./tests/data/mock_blurry_image.jpg",
        "invalid_image": "./tests/data/mock_invalid_image.jpg",
    }

@pytest.fixture
def mock_process_batch_data():
    """
    Fixture for simulating batch process image data.
    """
    return ["./tests/data/mock_image_1.jpg", "./tests/data/mock_image_2.jpg"]


# Evaluate image tests
@patch("app.evaluation.check_resolution")
@patch("app.evaluation.check_clarity")
@patch("app.evaluation.check_background")
def test_evaluate_image_good(mock_check_resolution, mock_check_clarity, mock_check_background, mock_image_paths):
    """
    Verify evaluate_image identifies a good image.
    """
    mock_check_resolution.return_value = (True, "Resolution OK")
    mock_check_clarity.return_value = (True, "Clarity OK", 120.0)
    mock_check_background.return_value = (True, "Background OK")

    result = evaluate_image(mock_image_paths["good_image"])
    assert result["status"] == "Good"
    assert result["issues"] == []

@patch("app.evaluation.check_resolution")
@patch("app.evaluation.check_clarity")
@patch("app.evaluation.check_background")
def test_evaluate_image_needs_improvement(mock_check_resolution, mock_check_clarity, mock_check_background, mock_image_paths):
    """
    Verify evaluate_image handles an image that needs improvements.
    """
    mock_check_resolution.return_value = (True, "Resolution OK")
    mock_check_clarity.return_value = (False, "Blurry", 50.0)
    mock_check_background.return_value = (False, "Complex background")

    result = evaluate_image(mock_image_paths["blurry_image"])
    assert result["status"] == "Needs Improvement"
    assert "Blurry" in result["issues"]
    assert "Complex background" in result["issues"]

def test_evaluate_image_invalid(mock_image_paths):
    """
    Verify evaluate_image raises an error for invalid images.
    """
    result = evaluate_image(mock_image_paths["invalid_image"])
    assert result["status"] == "Invalid Image"
    assert len(result["issues"]) > 0


# Batch processing test
@patch("batch_process.process_image")  # Assuming batch_process module has process_image
@patch("batch_process.ThreadPoolExecutor.map")  # Patch ThreadPoolExecutor's map method
def test_process_images_in_batches_parallel(mock_map, mock_process_image, mock_process_batch_data):
    """
    Validate parallel image batch processing using mocks.
    """
    mock_process_image.side_effect = lambda x: f"Processed {x}"
    mock_map.side_effect = lambda func, iterable: [func(x) for x in iterable]

    try:
        process_images_in_batches_parallel()  # Adjust call to use the defined process
        logging.info("Parallel batch processing executed successfully.")
        mock_map.assert_called_once()
        assert mock_process_image.call_count == len(mock_process_batch_data)
    except Exception as error:
        logging.error(f"Batch processing test failed: {error}")
        pytest.fail("Parallel batch image processing failed unexpectedly.")
