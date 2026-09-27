"""Shared, stateless feature preparation for training, Streamlit and the API."""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'log_income', 'age', 'age_sq', 'income_per_age',
    'employed_educated', 'education', 'work', 'car',
]


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the five raw fields to the model's eight ordered features.

    Income is not rescaled; the UI labels it as thousands of rubles.
    Callers validate input bounds before prediction; the bundled training rows
    contain positive ages and non-negative incomes.
    """
    features = data[['age', 'income', 'education', 'work', 'car']].astype(float).copy()
    features['log_income'] = np.log1p(features['income'])
    features['age_sq'] = features['age'] ** 2
    features['income_per_age'] = features['income'] / features['age']
    features['employed_educated'] = features['work'] * features['education']
    return features[FEATURE_NAMES]
