from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import service

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('age,income,education,work,car', [
    (27, 32.0, False, False, True),
    (50, 150.0, True, True, True),
])
def test_api_uses_the_same_features_and_decision_as_the_ui(
    age, income, education, work, car
):
    payload = dict(age=age, income=income, education=education, work=work, car=car)
    # The existing Streamlit form's eight-feature contract, independently specified.
    features = [[np.log1p(income), age, age**2, income / age,
                 int(work and education), int(education), int(work), int(car)]]
    expected_probability = joblib.load(ROOT / 'model.pkl').predict_proba(features)[0, 1]
    response = TestClient(service.app, raise_server_exceptions=False).post(
        '/score', json=payload
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['default_proba'] == pytest.approx(expected_probability)
    assert result['approved'] is bool(expected_probability < 0.5)


@pytest.mark.parametrize('field,value', [('age', 0), ('age', 101), ('income', -1)])
def test_api_rejects_values_outside_the_form_contract(field, value):
    payload = dict(age=27, income=32, education=False, work=False, car=True)
    payload[field] = value
    response = TestClient(service.app, raise_server_exceptions=False).post(
        '/score', json=payload
    )
    assert response.status_code == 422
