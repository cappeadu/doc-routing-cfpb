from contextlib import asynccontextmanager
from http import HTTPStatus

import pandas as pd
from fastapi import FastAPI, Request

from src.config import nlp
from src.evaluate import get_all_metrics
from src.predict import TFIDFPredictor, predict_with_proba

CHECKPOINT = "models/artifacts.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = TFIDFPredictor.from_checkpoint(CHECKPOINT)
    yield


app = FastAPI(
    title="Complaint Routing",
    description="Routes complaints to the appropriate Unit/Department",
    lifespan=lifespan,
    version="0.1",
)


@app.get("/")
def healthcheck():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/")
async def predict_(request: Request, complaint: str):
    predictor = request.app.state.predictor
    sentence = " ".join([token.lemma_ for token in nlp(complaint)])
    df = pd.DataFrame({"complaint_what_happened": [complaint], "text": [sentence]})
    results = predict_with_proba(df, predictor)
    return {"results": results}


@app.post("/evaluate/")
async def evaluate_(
    dataset_location: str, results_dir: str, checkpoint: str = CHECKPOINT
):
    results = get_all_metrics(
        dataset_loc=dataset_location, results_dir=results_dir, checkpoint=checkpoint
    )
    return {"results": results}
