from pathlib import Path

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from service import app

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_submits_the_same_input_as_the_api_and_keeps_threshold_control():
    ui = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=10).run()
    assert not ui.exception
    ui.number_input[0].set_value(27)
    ui.number_input[1].set_value(32.0)
    ui.checkbox[2].check()
    ui.button[0].click().run()
    assert not ui.exception
    response = TestClient(app).post('/score', json=dict(
        age=27, income=32, education=False, work=False, car=True
    )).json()
    assert any(f"{response['default_proba']:.1%}" in item.value for item in ui.markdown)
    assert bool(ui.success) is response['approved']
    assert len(ui.json) == 1

    # Changing the UI threshold must still change the decision independently.
    for threshold in (0.1, 0.9):
        ui.sidebar.slider[0].set_value(threshold)
        ui.button[0].click().run()
        assert not ui.exception
        assert bool(ui.success) is bool(response['default_proba'] < threshold)
