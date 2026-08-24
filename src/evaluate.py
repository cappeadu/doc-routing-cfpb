import datetime
import time
from collections import OrderedDict
from typing import Annotated

import numpy as np
import typer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.data import lemmatize_text, load_dataset
from src.predict import TFIDFPredictor
from src.utils import save_dict

app = typer.Typer()


def overall_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """Get overall metrics.

    Args:
        y_true (np.array): actual labels.
        y_pred (np.array): predicted labels.

    Returns:
        Dict: overall metrics
    """
    metrics = {}
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    eval_metrics = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="macro"
    )
    weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    metrics["accuracy"] = accuracy
    metrics["precision_macro"] = eval_metrics[0]
    metrics["recall_macro"] = eval_metrics[1]
    metrics["f1_macro"] = eval_metrics[2]
    metrics["f1_weighted"] = weighted_f1
    metrics["num_samples"] = np.float64(len(y_true))

    return metrics


def per_class_metrics(y_true: np.array, y_pred: np.array, class_to_idx: dict) -> dict:
    """Get per class metrics.

    Args:
        y_true (np.array): actual labels.
        y_pred (np.array): predicted labels.
        class_to_idx (dict): mapping of labels to index.

    Returns:
        Dict: per class metrics
    """
    metrics = {}
    per_class_metrics = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=None
    )
    for idx, class_ in enumerate(class_to_idx):
        metrics[class_] = {
            "precision": per_class_metrics[0][idx],
            "recall": per_class_metrics[1][idx],
            "f1": per_class_metrics[2][idx],
            "num_samples": np.float64(per_class_metrics[3][idx]),
        }
    sorted_metrics = OrderedDict(
        sorted(metrics.items(), key=lambda label: label[1]["f1"], reverse=True)
    )
    return sorted_metrics


@app.command()
def evaluate(
    dataset_loc: Annotated[
        str, typer.Option(help="location of dataset for evaluation.")
    ],
    checkpoint: Annotated[str, typer.Option(help="location of checkpoint.")],
    results_dir: Annotated[
        str, typer.Option(help="location to save metrics after evals.")
    ] = "results",
) -> dict:
    """Get all metrics for dataset.

    Args:
        y_true (np.array): actual labels.
        y_pred (np.array): predicted labels.
        class_to_idx (dict): mapping of labels to index.

    Returns:
        Dict: metrics for dataset.
    """
    start_time = time.time()
    predictor = TFIDFPredictor.from_checkpoint(checkpoint)
    df = load_dataset(dataset_loc)
    class_to_idx = predictor.preprocessor.class_to_idx
    labels_list = df["labels"].tolist()
    y_true = [class_to_idx[label] for label in labels_list]
    df["text"] = lemmatize_text(df)
    y_pred = predictor(df)
    end_time = time.time()
    metrics = {
        "time stamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "total time": f"{(end_time - start_time) / 60:.2f} mins",
        "overall": overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": per_class_metrics(
            y_true=y_true, y_pred=y_pred, class_to_idx=class_to_idx
        ),
    }

    if results_dir:
        save_dict(
            directory=results_dir,
            dict_to_save=metrics,
        )

    return metrics


if __name__ == "__main__":
    app()
