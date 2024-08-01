import pytest
import json
from app import app  # Correct import statement


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    # Sample test data
    test_data = {
        'features': [5.1, 3.5, 1.4, 0.2]  # Example features
    }

    # Make a POST request to the /predict endpoint
    response = client.post('/predict', data=json.dumps(test_data),
                           content_type='application/json')

    # Verify the response
    assert response.status_code == 200
    response_data = response.get_json()
    assert 'prediction' in response_data
    assert isinstance(response_data['prediction'], int)
