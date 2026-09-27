from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import prepare_features


class ClientData(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0, allow_inf_nan=False)
    education: bool
    work: bool
    car: bool


app = FastAPI(title='Credit scoring prototype')
model = joblib.load(Path(__file__).resolve().with_name('model.pkl'))


@app.post('/score')
def score(data: ClientData):
    features = prepare_features(pd.DataFrame([data.model_dump()]))
    default_proba = float(model.predict_proba(features.to_numpy())[0, 1])
    # Class 1 means default, so it must not be returned as approval.
    return {'approved': default_proba < 0.5, 'default_proba': default_proba}
