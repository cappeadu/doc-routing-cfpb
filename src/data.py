import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from src.config import STOPWORDS, nlp


def load_dataset(dataset_loc: str, num_samples=None) -> pd.DataFrame:
    """Load dataset into a dataframe.

    Args:
        dataset_loc (str): Location of dataset.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        pd.DataFrame: the loaded dataset.
    """
    df = pd.read_csv(dataset_loc)
    df = df.sample(num_samples, random_state=42) if num_samples else df
    return df


def clean_text(text: str, stopwords: set = STOPWORDS) -> str:
    """Clean text.

    Args:
        text (str): Text to clean.
        stopwords (set): Used to filter stop words. Defaults to STOPWORDS.

    Returns:
        str: the cleaned text.
    """
    text = text.lower()
    # filtering out stop words
    text = " ".join([token for token in text.split() if token not in stopwords])
    # remove punctuations
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    # remove non-alphanumeric and numeric characters
    text = re.sub("[^A-Za-z]+", " ", text)
    pattern = re.compile(r"(xx+\s*)+")  # remove repeated XXXX characters
    text = pattern.sub(" mask ", text)  # rename repeated XXXX characters to 'mask'
    text = re.sub(" +", " ", text).strip()  # remove repeated spaces
    text = re.sub(r"http\S+", "", text)  # remove hyperlinks
    return text


def lemmatize_text(df: pd.DataFrame) -> list[str]:
    """Lemmatize words in text.

    Args:
        df (pd.DataFrame): Dataset to be lemmatized.

    Returns:
        list[str]: List of sentences/words that have been lemmatized. Same size as df.
    """
    all_complaints = df["complaint_what_happened"].tolist()
    complaints_docs = nlp.pipe(all_complaints, batch_size=128, n_process=-1)

    lemmatized_sentences = []
    for doc in tqdm(complaints_docs, desc="Lemmatizing.."):
        lemmatized_sentences.append(" ".join([token.lemma_ for token in doc]))

    return lemmatized_sentences


def preprocess_dataset(dataframe: pd.DataFrame, class_to_idx: dict) -> pd.DataFrame:
    """Preprocess dataset.

    Args:
        dataframe (pd.DataFrame): Dataset to prepocess.
        class_to_index(dict): Mapping of label name to label id.

    Returns:
        pd.DataFrame: preprocessed dataset.
    """
    df = dataframe.copy()
    tqdm.pandas(desc="Processing texts", colour="green")

    df["text"] = df.text.progress_apply(clean_text)
    if "labels" in df.columns:
        df["labels"] = df.labels.map(class_to_idx)
        return df[["product", "complaint_what_happened", "text", "labels"]]
    else:
        return df[["complaint_what_happened", "text"]]


# Uses the preprocess_dataset function
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor class"""

    def fit(self, df: pd.DataFrame):
        tags = sorted(df["labels"].unique())  # sort tag names for reproducibility
        self.class_to_idx = {label: idx for idx, label in enumerate(tags)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        return self

    def transform(self, df: pd.DataFrame, y=None):
        return preprocess_dataset(df, class_to_idx=self.class_to_idx)
