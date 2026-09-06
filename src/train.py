import datetime
import json
from typing import Annotated

import typer

from src.config import logger
from src.data import CustomPreprocessor, lemmatize_text, load_dataset
from src.models import LogisticRegression, TFIDFTrainer, TfidfVectorizer

app = typer.Typer()


@app.command()
def train_model(
    train_data_loc: Annotated[
        str, typer.Option(help="path location of train dataset.")
    ],
    val_data_loc: Annotated[
        str, typer.Option(help="path location of validation dataset.")
    ],
    model_params: Annotated[str, typer.Option(help="model parameters.")],
    vectorizer_params: Annotated[
        str, typer.Option(help="parameters for TF-IDF vectorizer.")
    ],
    experiment_name: Annotated[str, typer.Option(help="name of MLFLOW experiment.")],
    directory_to_save_model: Annotated[
        str, typer.Option(help="directory to save models.")
    ],
    log_experiment: Annotated[
        bool, typer.Option(help="log experiment with MLFLOW")
    ] = True,
) -> LogisticRegression:
    """Training function to train the Logistic model using TF-IDF vectorizer.

    Args:
        train_data_loc (str): path location of train dataset.
        val_data_loc (str): path location of validation dataset.
        model_params (dict): model parameters.
        vectorizer_params (dict): parameters for TF-IDF vectorizer.
        experiment_name (str): name of MLFLOW experiment.
        directory (str): directory to save models.
        log_experiment (bool): log experiment with MLFLOW. Defaults to True.

    Returns:
        sklearn.linear_model.LogisticRegression
    """
    train_df = load_dataset(train_data_loc)
    val_df = load_dataset(val_data_loc)

    train_df["text"] = lemmatize_text(train_df)
    val_df["text"] = lemmatize_text(val_df)

    train_df_copied = train_df.copy()
    val_df_copied = val_df.copy()

    vectorizer_params = json.loads(vectorizer_params)
    vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    model_params = json.loads(model_params)

    trainer = TFIDFTrainer(
        model=LogisticRegression,
        preprocessor=CustomPreprocessor(),
        vectorizer=TfidfVectorizer,
        train_dataset=train_df_copied,
        val_dataset=val_df_copied,
        directory=directory_to_save_model,
        vectorizer_params=vectorizer_params,
        params=model_params,
        log_experiment=log_experiment,
        experiment_name=experiment_name,
    )
    model_artifact = trainer.train()
    logs = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "experiment_name": experiment_name,
    }
    logger.info(json.dumps(logs))

    return model_artifact


if __name__ == "__main__":
    app()
