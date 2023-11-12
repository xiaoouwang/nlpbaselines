import random
import numpy as np
import pandas as pd

def data_sentiment(n=30, split=0.8):
    """
    Generate a pandas DataFrame with n rows of random sentences, labels, and split information.

    Args:
    - n (int): number of sentences to generate
    - split (float): proportion of data to use for training, default is 0.8

    Returns:
    - df (pandas.DataFrame): DataFrame with columns "text", "label", and "split"
    """

    data = {
        "text": [f"This is sentence {i}" for i in range(n)],
        "label": [random.randint(0, 1) for _ in range(n)],
        "split": ["train" if i < int(split*n) else "validation" for i in range(n)],
    }
    df = pd.DataFrame(data)
    return df