from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app
import pytest
import pandas as pd

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

@patch("src.api.main.requests.post")
def test_classify_image_mock(mock_post):
    # Mocking the External Model Server response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [
            {"predicted_class": "Borobudur Temple", "confidence": 0.98}
        ]
    }
    mock_post.return_value = mock_response

    # Create dummy image bytes
    dummy_image = b"fake_image_bytes"
    
    response = client.post(
        "/predict/classify", 
        files={"file": ("test.jpg", dummy_image, "image/jpeg")}
    )
    
    # Assertions
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["predicted_class"] == "Borobudur Temple"
    assert json_resp["confidence"] == 0.98
    
    # Verify we sent base64
    assert mock_post.called
    sent_payload = mock_post.call_args[1]['json']
    assert "inputs" in sent_payload
    assert len(sent_payload["inputs"]) == 1
    assert isinstance(sent_payload["inputs"][0], str) # Should be base64 string

@patch("src.api.main.recommender_model")
def test_recommend_places_mock(mock_recommender):
    # Mocking the Global Recommender Model
    # It returns (DataFrame, error_string)
    mock_df = pd.DataFrame([
        {
            "Place_Id": 101,
            "Place_Name": "Monas",
            "City": "Jakarta", 
            "Category": "Budaya",
            "Rating": 4.5,
            "Price": 15000,
            "Similarity": 0.95
        }
    ])
    
    mock_recommender.recommend.return_value = (mock_df, None)
    
    response = client.post(
        "/predict/recommend", 
        json={"place_name": "Old City", "top_n": 1}
    )
    
    assert response.status_code == 200
    json_resp = response.json()
    assert len(json_resp["recommendations"]) == 1
    assert json_resp["recommendations"][0]["Place_Name"] == "Monas"
