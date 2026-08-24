import time

import joblib
from sentence_transformers import SentenceTransformer

from src_embeddings.data import clean_text_for_embeddings


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
    text: str, predictor: EmbeddingsPredictor, class_to_idx: list[str]
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
        result.append(
            {
                "prediction": class_to_idx[pred],
                "probabilities": sorted_probs,
                "latency": f"{(end_time - start_time) * 1000:.2f} ms",
            }
        )

    return result
