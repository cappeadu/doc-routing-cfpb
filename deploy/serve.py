from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.config import nlp
from src.evaluate import evaluate
from src.predict import TFIDFPredictor, predict_with_proba
from src_bert.predict import BERTPredictor, predict_with_proba_bert
from src_embeddings.predict import EmbeddingsPredictor, predict_with_proba_embeddings

CHECKPOINT_TFIDF = "models/artifacts.joblib"
CHECKPOINT_EMBEDDINGS = "models/embeddings_model.joblib"
CHECKPOINT_BERT = "appcle/distilbert-base-uncased-cfpd"


class ModelChoice(str, Enum):
    tfidf = "tfidf"
    embeddings = "embeddings"
    bert = "bert"


class Complaint(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tfidf_predictor = TFIDFPredictor.from_checkpoint(CHECKPOINT_TFIDF)
    app.state.embeddings_predictor = EmbeddingsPredictor.from_checkpoint(
        CHECKPOINT_EMBEDDINGS
    )
    app.state.bert = BERTPredictor(CHECKPOINT_BERT)
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


@app.post("/predict")
async def predict_(
    complaint: Complaint,
    request: Request,
    model: ModelChoice = ModelChoice.tfidf,
    threshold: float = 0.7,
):
    if model.value == "tfidf":
        sentence = " ".join([token.lemma_ for token in nlp(complaint.text)])
        df = pd.DataFrame(
            {"complaint_what_happened": [complaint.text], "text": [sentence]}
        )
        predictor = request.app.state.tfidf_predictor
        results = predict_with_proba(df, predictor, threshold)

    elif model.value == "embeddings":
        class_to_idx = request.app.state.tfidf_predictor.preprocessor.class_to_idx
        predictor = request.app.state.embeddings_predictor
        results = predict_with_proba_embeddings(
            complaint.text, predictor, list(class_to_idx), threshold
        )
    elif model.value == "bert":
        predictor = request.app.state.bert
        results = predict_with_proba_bert(
            complaint.text, predictor=predictor, threshold=threshold
        )

    else:
        raise HTTPException(status_code=404, detail="Unknown Model")

    return {"results": results, "model": model.value}


@app.post("/evaluate/")
async def evaluate_(
    dataset_location: str,
    results_dir: None,
    checkpoint: str = CHECKPOINT_TFIDF,
):
    results = evaluate(
        dataset_loc=dataset_location, results_dir=results_dir, checkpoint=checkpoint
    )
    return {"results": results}
