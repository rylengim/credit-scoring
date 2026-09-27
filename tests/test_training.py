from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection

ROOT = Path(__file__).resolve().parents[1]


def test_model_selection_never_uses_the_held_out_rows(monkeypatch, tmp_path):
    data = pd.read_csv(ROOT / 'data/scoring_features.csv').sample(400, random_state=7)
    monkeypatch.setattr(pd, 'read_csv', lambda *args, **kwargs: data.copy())
    original_split = model_selection.train_test_split
    original_dump = joblib.dump
    split = {}
    selection_inputs = []

    def record_split(*args, **kwargs):
        result = original_split(*args, **kwargs)
        split['train_x'], split['test_x'], split['train_y'], _ = result
        return result

    def record_selection(model, x, y, **kwargs):
        selection_inputs.append((np.asarray(x).copy(), np.asarray(y).copy()))
        # Avoid expensive five-fold fitting: this test checks the split boundary.
        return np.array([0.7 if len(selection_inputs) == 1 else 0.6] * 5)

    monkeypatch.setattr(model_selection, 'train_test_split', record_split)
    monkeypatch.setattr(model_selection, 'cross_val_score', record_selection)
    monkeypatch.setattr(joblib, 'dump', lambda model, path: original_dump(model, tmp_path / 'model.pkl'))
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')

    assert len(selection_inputs) == 2
    for x, y in selection_inputs:
        np.testing.assert_array_equal(x, split['train_x'])
        np.testing.assert_array_equal(y, split['train_y'])
    assert len(split['test_x']) == 80
