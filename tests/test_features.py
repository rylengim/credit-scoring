from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, prepare_features

ROOT = Path(__file__).resolve().parents[1]


def test_shared_preparation_reproduces_the_existing_training_features():
    raw = pd.read_csv(ROOT / 'data/scoring.csv')
    expected = pd.read_csv(ROOT / 'data/scoring_features.csv')
    actual = prepare_features(raw)
    assert actual.columns.tolist() == FEATURE_NAMES
    np.testing.assert_allclose(actual, expected[FEATURE_NAMES], rtol=1e-12, atol=1e-12)


def test_zero_income_is_a_valid_finite_form_input():
    actual = prepare_features(pd.DataFrame([dict(
        age=18, income=0, education=False, work=False, car=False
    )]))
    np.testing.assert_array_equal(actual.to_numpy(), [[0, 18, 324, 0, 0, 0, 0, 0]])
