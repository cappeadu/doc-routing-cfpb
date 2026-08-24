import re

import numpy as np


def load_dataset(dataset_loc: str) -> np.array:
    """Load embeddings.

    Args:
        dataset_loc (str): Location of dataset.

    Returns:
        np.array: the loaded dataset.
    """
    with open(dataset_loc, "rb") as f:
        X = np.load(f)
        y = np.load(f)
    return X, y


def clean_text_for_embeddings(sentence: str) -> str:
    """Clean text.

    Args:
        sentence (str): Text to clean.

    Returns:
        str: the cleaned text.
    """
    sentence = sentence.lower()
    # remove punctuations
    sentence = re.sub(r"([!\"'#$%&()*\+-/:;<=>?@\\\[\]^_`{|}~])", r" \1 ", sentence)
    # remove non-alphanumeric.
    sentence = re.sub("[^A-Za-z0-9]+", " ", sentence)
    # remove repeated XXXX characters
    pattern = re.compile(r"(xx+\s*)+")
    # rename repeated XXXX characters to 'mask'
    sentence = pattern.sub("mask ", sentence)
    sentence = re.sub(" +", " ", sentence).strip()  # remove repeated spaces
    sentence = " ".join(sentence.split())
    sentence = re.sub(r"http\S+", "", sentence)  # remove hyperlinks
    return sentence
