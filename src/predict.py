import joblib
import pandas as pd

from src.config import nlp


class TFIDFPredictor:
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


# predict proba
def predict_with_proba(df, predictor):
    preprocessor = predictor.preprocessor
    vectorizer = predictor.vectorizer
    model = predictor.model

    df_transformed = preprocessor.transform(df)
    df_vectorized = vectorizer.transform(df_transformed.text).toarray()
    predicted_probas = model.predict_proba(df_vectorized)

    labels = [key for key in preprocessor.class_to_idx]

    all_pred_with_proba = []
    for score in predicted_probas:
        pred = int(score.argmax())
        all_pred_with_proba.append(
            {
                "prediction": labels[pred],
                "probabilities": {a: round(float(b), 4) for a, b in zip(labels, score)},
            }
        )

    return all_pred_with_proba
