import os
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

load_dotenv()


class EmbeddingsClassifier:
    """Class to train Logistic Model using embeddings"""

    def __init__(
        self,
        model: LogisticRegression,
        directory: str,
        train_dataset: tuple[np.array, np.array],
        val_dataset: tuple[np.array, np.array],
        experiment_name=None,
        log_experiment: bool = False,
        params=None,
    ):
        if params == None:
            self.params = None
            self.model = model()
        else:
            self.params = params
            self.model = model(**self.params)  # i will save this

        self.directory_name = directory  # path i will use to save above
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.experiment_name = experiment_name
        self.log_experiment = log_experiment

    # save trained artifacts
    def save_artifacts(self, model):
        artifacts = {"model": model}
        artifacts_path = Path(__name__).parent.parent / self.directory_name
        if not artifacts_path.exists():
            artifacts_path.mkdir(parents=True)
        joblib.dump(artifacts, artifacts_path / "embeddings_model.joblib")

    # to track experiment
    def mlflow_logging(self, train_embeddings, train_labels):
        mlflow.set_tracking_uri(os.environ.get("ML_FLOW_TRACKING_URI"))
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            self.model.fit(train_embeddings, train_labels)
            y_train_pred = self.model.predict(train_embeddings)

            X_val, y_val = self.val_dataset[0], self.val_dataset[1]
            y_val_pred = self.model.predict(X_val)

            for name, y_true, y_pred in [
                ("train", train_labels, y_train_pred),
                ("val", y_val, y_val_pred),
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

    # train function to perform training
    def train(self):
        start = time.time()
        X_train, y_train = self.train_dataset[0], self.train_dataset[1]
        print(f"Train Examples: {X_train.shape[0]}")

        if self.log_experiment:
            print(
                f"Training and Logging to MLFLOW experiment {self.experiment_name}..."
            )
            self.mlflow_logging(train_embeddings=X_train, train_labels=y_train)
        else:
            print("Training...")
            self.model.fit(X_train, y_train)

        self.save_artifacts(model=self.model)
        end = time.time()
        total_time = end - start
        print(f"Total training time {total_time:.2f} secs")
        return self.model
