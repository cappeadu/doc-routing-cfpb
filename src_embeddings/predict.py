import json
import time
from typing import Annotated

import joblib
import typer
from sentence_transformers import SentenceTransformer

from src.config import logger
from src.predict import TFIDFPredictor
from src_embeddings.data import clean_text_for_embeddings

app = typer.Typer()


class EmbeddingsPredictor:
    """Class to load artifacts and perform predictions."""

    def __init__(self, model):
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint: str):
        artifacts = joblib.load(checkpoint)
        model = artifacts["model"]
        return cls(model)

    def __call__(self, texts: list[str]):
        embeddings = self.embeddings_model.encode(texts)
        return self.model.predict(embeddings)

    def predict_embeddings_only(self, embeddings):
        return self.model.predict(embeddings)


# To produce predictions with probabilities(probabilities are sorted)
def predict_with_proba_embeddings(
    text: str,
    predictor: EmbeddingsPredictor,
    class_to_idx: list[str],
    threshold: float = 0.7,
) -> list[dict]:
    """Perform predictions and show probabilities associated with predictions.

    Args:
        text (str): Examples to be predicted.
        predictor (EmbeddingsPredictor): Predictor class to perform predictions.
        class_to_idx (list[str]): Label names with matches same as used during training.

    Returns:
        list[dict]: list of predictions.

    """
    start_time = time.time()
    cleaned_text = clean_text_for_embeddings(text)
    dense_embeddings = predictor.embeddings_model.encode([cleaned_text])
    predicted_probas = predictor.model.predict_proba(dense_embeddings)
    end_time = time.time()

    result = []
    for scores in predicted_probas:
        pred = int(scores.argmax())
        all_probs = {
            label: round(float(score), 4) for label, score in zip(class_to_idx, scores)
        }
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        prob = max(scores)
        decision = "auto_route" if prob >= threshold else "human_review"
        result.append(
            {
                "prediction": class_to_idx[pred],
                "probabilities": sorted_probs,
                "latency": f"{(end_time - start_time) * 1000:.2f} ms",
                "decision": decision,
            }
        )

    return result


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

    CHECKPOINT_EMBEDDINGS = "models/embeddings_model.joblib"
    CHECKPOINT_TFIDF = "models/artifacts.joblib"
    class_to_idx = TFIDFPredictor.from_checkpoint(
        CHECKPOINT_TFIDF
    ).preprocessor.class_to_idx
    class_to_idx = list(class_to_idx)
    predictor = EmbeddingsPredictor.from_checkpoint(CHECKPOINT_EMBEDDINGS)
    results = predict_with_proba_embeddings(text, predictor, class_to_idx)
    print(results)
    logger.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    app()
