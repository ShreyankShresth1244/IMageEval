import pytest
from unittest.mock import patch
from app.api import app

@pytest.fixture
def client():
    """
    Create a test client for the Flask app.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@patch("app.utils.download_image")
def test_evaluate_endpoint(mock_download_image, client):
    """
    Test the /evaluate endpoint with a valid image URL.
    """
    # Mock the download_image function to avoid actual HTTP requests
    mock_download_image.return_value = None

    response = client.post("/evaluate", json={"image_url": "https://example.com/test.jpg"})
    assert response.status_code == 200
    assert "status" in response.json
    assert response.json["status"] in ["Good", "Needs Improvement"]

@patch("app.utils.download_image")
def test_enhance_endpoint(mock_download_image, client):
    """
    Test the /enhance endpoint with a valid image URL.
    """
    # Mock the download_image function to avoid actual HTTP requests
    mock_download_image.return_value = None

    response = client.post("/enhance", json={"image_url": "https://example.com/test.jpg"})
    assert response.status_code == 200
    assert "enhanced_image_path" in response.json
    assert response.json["enhanced_image_path"].endswith(".jpg")

def test_evaluate_endpoint_missing_url(client):
    """
    Test the /evaluate endpoint with a missing image URL.
    """
    response = client.post("/evaluate", json={})
    assert response.status_code == 400
    assert "error" in response.json

def test_enhance_endpoint_invalid_url(client):
    """
    Test the /enhance endpoint with an invalid image URL.
    """
    response = client.post("/enhance", json={"image_url": "invalid-url"})
    assert response.status_code == 400
    assert "error" in response.json
