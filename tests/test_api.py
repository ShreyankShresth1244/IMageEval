import pytest
from unittest.mock import patch, MagicMock
from app import app  # Ensure your Flask app is imported correctly

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
    # Mock the download_image utility function
    mock_image = MagicMock()
    mock_download_image.return_value = mock_image

    response = client.post(
        "/evaluate", json={"image_url": "https://example.com/sample.jpg"}
    )

    assert response.status_code == 200
    assert "status" in response.json
    assert response.json["status"] in ["Good", "Needs Improvement"]

@patch("app.utils.download_image")
def test_enhance_endpoint(mock_download_image, client):
    """
    Test the /enhance endpoint with a valid image URL.
    """
    # Mock the download_image utility function
    mock_image = MagicMock()
    mock_download_image.return_value = mock_image

    # Simulate successful image enhancement
    mock_image.save = MagicMock()

    response = client.post(
        "/enhance", json={"image_url": "https://example.com/sample.jpg"}
    )

    assert response.status_code == 200
    assert "enhanced_image_path" in response.json
    assert response.json["enhanced_image_path"].endswith(".jpg")

# Ensure additional setup for other Flask routes is correct
if __name__ == "__main__":
    pytest.main()
