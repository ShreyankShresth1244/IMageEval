import os
import pytest
from unittest.mock import patch
from app.evaluation import evaluate_image

@pytest.fixture
def test_image_paths():
    """
    Fixture for providing test image paths.
    """
    return {
        "good_image": "./tests/data/test_good.jpg",
        "blurry_image": "./tests/data/test_blurry.jpg",
        "invalid_image": "./tests/data/invalid.jpg",
    }

@patch("app.evaluation.check_resolution")
@patch("app.evaluation.check_clarity")
@patch("app.evaluation.check_background")
def test_evaluate_image_good(mock_background, mock_clarity, mock_resolution, test_image_paths):
    """
    Test evaluate_image with a good image.
    """
    # Mock the checks to simulate a "Good" image
    mock_resolution.return_value = (True, "Resolution OK")
    mock_clarity.return_value = (True, "Clarity OK")
    mock_background.return_value = (True, "Background OK")

    result = evaluate_image(test_image_paths["good_image"])
    assert result["status"] == "Good"
    assert result["issues"] == []

@patch("app.evaluation.check_resolution")
@patch("app.evaluation.check_clarity")
@patch("app.evaluation.check_background")
def test_evaluate_image_needs_improvement(mock_background, mock_clarity, mock_resolution, test_image_paths):
    """
    Test evaluate_image with an image needing improvement.
    """
    # Mock the checks to simulate a "Needs Improvement" image
    mock_resolution.return_value = (True, "Resolution OK")
    mock_clarity.return_value = (False, "Blurry")
    mock_background.return_value = (False, "Complex background")

    result = evaluate_image(test_image_paths["blurry_image"])
    assert result["status"] == "Needs Improvement"
    assert "Blurry" in result["issues"]
    assert "Complex background" in result["issues"]

def test_evaluate_image_invalid(test_image_paths):
    """
    Test evaluate_image with an invalid image.
    """
    result = evaluate_image(test_image_paths["invalid_image"])
    assert result["status"] == "Invalid Image"
    assert "Cannot load image" in result["issues"]
