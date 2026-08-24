import json
from typing import Annotated

import typer

from src_embeddings.data import load_dataset
from src_embeddings.models import EmbeddingsClassifier, LogisticRegression

app = typer.Typer()


@app.command()
def train_model(
    train_data_loc: Annotated[str, typer.Option(help="path location of train dataset")],
    val_data_loc: Annotated[
        str, typer.Option(help="path location of validation dataset")
    ],
    model_params: Annotated[str, typer.Option(help="model parameters.")],
    experiment_name: Annotated[str, typer.Option(help="name of MLFLOW experiment.")],
    directory_to_save_model: Annotated[
        str, typer.Option(help="directory to save model.")
    ],
    log_experiment: Annotated[
        bool, typer.Option(help="log experiment with MLFLOW")
    ] = True,
) -> LogisticRegression:
    """Training function to train the Logistic model using embeddings.

    Args:
        train_data_loc (str): path location of train dataset.
        val_data_loc (str): path location of validation dataset.
        model_params (dict): model parameters.
        experiment_name (str): name of MLFLOW experiment.
        directory_to_save_model (str): directory to save models.
        log_experiment (bool): log experiment with MLFLOW. Defaults to True.

    Returns:
        sklearn.linear_model.LogisticRegression
    """
    X_train, y_train = load_dataset(train_data_loc)
    X_val, y_val = load_dataset(val_data_loc)
    model_params = json.loads(model_params)

    trainer = EmbeddingsClassifier(
        model=LogisticRegression,
        train_dataset=(X_train, y_train),
        val_dataset=(X_val, y_val),
        directory=directory_to_save_model,
        params=model_params,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
    )

    return trainer.train()


if __name__ == "__main__":
    app()
