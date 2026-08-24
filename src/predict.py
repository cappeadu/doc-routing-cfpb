import time

import joblib
import pandas as pd


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
def predict_with_proba(df: pd.DataFrame, predictor: TFIDFPredictor) -> list[dict]:
    """Perform predictions and show probabilities associated with predictions.

    Args:
        df (pd.DataFrame): Examples to be predicted.
        predictor (TFIDFPredictor): Predictor class to perform predictions.

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
        all_pred_with_proba.append(
            {
                "prediction": labels[pred],
                "probabilities": sorted_probs,
                "latency": f"{(end_time - start_time) * 1000:.2f} ms",
            }
        )

    return all_pred_with_proba
