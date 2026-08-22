import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.config import mlflow
from src.data import CustomPreprocessor


class TFIDFTrainer:
    def __init__(
        self,
        model: LogisticRegression,
        preprocessor: CustomPreprocessor,
        vectorizer: TfidfVectorizer,
        directory: str,
        train_dataset: pd.DataFrame,
        val_dataset: pd.DataFrame,
        vectorizer_params: dict,
        experiment_name=None,
        log_experiment: bool = False,
        params=None,
    ):
        if params == None:
            self.params = None
            self.model = model()
        else:
            self.params = params
            self.model = model(**self.params)

        self.vectorizer_params = vectorizer_params
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer(**self.vectorizer_params)
        self.directory_name = directory
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.experiment_name = experiment_name
        self.log_experiment = log_experiment

    def preprocess(self):
        dataset = self.preprocessor.fit_transform(self.train_dataset)
        dataset["text_length"] = dataset.text.apply(lambda x: len(x.split()))
        short_complaints = dataset["text_length"] <= 3
        return dataset[~short_complaints]

    def save_artifacts(self, preprocessor, vectorizer, model):
        artifacts = {
            "model": model,
            "preprocessor": preprocessor,
            "vectorizer": vectorizer,
        }
        artifacts_path = Path(__file__).parent.parent / self.directory_name
        if not artifacts_path.exists():
            artifacts_path.mkdir(parents=True)
        joblib.dump(artifacts, artifacts_path / "artifacts.joblib")

    def mlflow_logging(self, train_vectors, train_labels):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            self.model.fit(train_vectors, train_labels)
            y_train_pred = self.model.predict(train_vectors)

            val_dataset_processed = self.preprocessor.transform(self.val_dataset)
            val_vectors = self.vectorizer.transform(
                val_dataset_processed.text
            ).toarray()
            y_val_pred = self.model.predict(val_vectors)
            val_labels = val_dataset_processed.labels

            for name, y_true, y_pred in [
                ("train", train_labels, y_train_pred),
                ("val", val_labels, y_val_pred),
            ]:
                metrics = precision_recall_fscore_support(
                    y_true=y_true, y_pred=y_pred, average="macro"
                )
                mlflow.log_metrics(
                    {
                        f"acc_{name}": accuracy_score(y_true=y_true, y_pred=y_pred),
                        f"prec_macro_{name}": metrics[0],
                        f"recall_macro_{name}": metrics[1],
                        f"f1_macro_{name}": metrics[2],
                        f"weighted_f1_{name}": f1_score(
                            y_true=y_true, y_pred=y_pred, average="weighted"
                        ),
                    }
                )
            if self.params != None:
                mlflow.log_params(self.params)

    def train(self):
        start = time.time()
        preprocessed_dataset = self.preprocess()
        print(f"Train Examples: {preprocessed_dataset.shape[0]}")
        labels = preprocessed_dataset.labels
        train_vectors = self.vectorizer.fit_transform(
            preprocessed_dataset["text"]
        ).toarray()

        if self.log_experiment:
            print(
                f"Training and Logging to MLFLOW experiment {self.experiment_name}..."
            )
            self.mlflow_logging(train_vectors=train_vectors, train_labels=labels)
        else:
            print("Training...")
            self.model.fit(train_vectors, labels)

        self.save_artifacts(
            preprocessor=self.preprocessor, vectorizer=self.vectorizer, model=self.model
        )
        end = time.time()
        total_time = end - start
        print(f"Total training time {total_time:.2f} secs")
