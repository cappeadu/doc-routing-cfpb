import json
import time
from typing import Annotated

import joblib
import pandas as pd
import typer

from src.config import logger, nlp

app = typer.Typer()


class TFIDFPredictor:
    """Class to load artifacts and perform predictions."""

    def __init__(self, preprocessor, vectorizer, model):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint: str):
        artifacts = joblib.load(checkpoint)
        preprocessor = artifacts["preprocessor"]
        vectorizer = artifacts["vectorizer"]
        model = artifacts["model"]
        return cls(preprocessor, vectorizer, model)

    def __call__(self, dataset: pd.DataFrame):
        cleaned_dataset = self.preprocessor.transform(dataset)
        vectors = self.vectorizer.transform(cleaned_dataset.text).toarray()
        return self.model.predict(vectors)


# To produce predictions with probabilities(probabilities are sorted)
def predict_with_proba(
    df: pd.DataFrame, predictor: TFIDFPredictor, threshold: float = 0.7
) -> list[dict]:
    """Perform predictions and show probabilities associated with predictions.

    Args:
        df (pd.DataFrame): Examples to be predicted.
        predictor (TFIDFPredictor): Predictor class to perform predictions.
        threshold (float): Prediction propabilities above this threshold are automatically routed.

    Returns:
        list[dict]: list of predictions.

    """
    start_time = time.time()
    preprocessor = predictor.preprocessor
    vectorizer = predictor.vectorizer
    model = predictor.model

    df_transformed = preprocessor.transform(df)
    df_vectorized = vectorizer.transform(df_transformed.text).toarray()
    predicted_probas = model.predict_proba(df_vectorized)
    end_time = time.time()

    labels = [key for key in preprocessor.class_to_idx]

    all_pred_with_proba = []
    for score in predicted_probas:
        pred = int(score.argmax())
        all_probs = {
            label: round(float(score_), 4) for label, score_ in zip(labels, score)
        }
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        prob = max(score)
        decision = "auto_route" if prob >= threshold else "human_review"
        all_pred_with_proba.append(
            {
                "prediction": labels[pred],
                "probabilities": sorted_probs,
                "latency": f"{(end_time - start_time) * 1000:.2f} ms",
                "decision": decision,
            }
        )

    return all_pred_with_proba


@app.command()
def predict_text_cli(
    text: Annotated[str, typer.Option(help="complaint text")],
) -> list[dict]:
    """Predict complaint text into appropriate department/unit.

    Args:
        text (str): complaint text to predict.

    Returns:
        list[dict]: Prediction result.
    """
    CHECKPOINT_TFIDF = "models/artifacts.joblib"
    predictor = TFIDFPredictor.from_checkpoint(CHECKPOINT_TFIDF)
    sentence = " ".join([token.lemma_ for token in nlp(text)])
    df = pd.DataFrame({"complaint_what_happened": [text], "text": [sentence]})
    results = predict_with_proba(df, predictor)
    print(results)
    logger.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    app()
