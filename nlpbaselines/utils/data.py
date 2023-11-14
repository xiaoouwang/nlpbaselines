import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from transformers import AutoTokenizer



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


class DatasetLoader:
    """
    A class for loading and encoding datasets for NLP tasks.

    Args:
        df_type (str): The type of the input dataframe. Default is "pandas".
        text_col (str): The name of the column containing the text data. Default is "text".
        label_col (str): The name of the column containing the label data. Default is "label".

    Methods:
        load_dataset(dataframe): Loads a dataset from a pandas DataFrame and returns training and validation datasets.
        load_dataset_cv(dataframe, n_splits): Loads a dataset from a pandas DataFrame and returns a list of training and validation datasets for cross-validation.
        encode_dataset(df, model_name): Encodes a dataset using a specified pre-trained tokenizer.

    """
class DatasetLoader:

    def __init__(self, df_type="pandas", text_col="text", label_col="label"):
        self.text_col = text_col
        self.label_col = label_col
        self.df_type = df_type

    def load_dataset(self, dataframe):
        # Split the DataFrame into training and validation sets
        dataframe.rename(columns={self.text_col: "text"}, inplace=True)
        train_df = dataframe[dataframe["split"] == "train"]
        val_df = dataframe[dataframe["split"] == "validation"]
        # get the number of unique labels
        num_labels = len(train_df[self.label_col].unique())
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        print(
            f"\nDataset loaded. The dataset has {num_labels} labels, {len(train_df)} training items, {len(val_df)} validation items. \n{dataframe.head(3)}"
        )
        return train_dataset, val_dataset

    def load_dataset_cv(self, dataframe, n_splits=5):
        dataframe.rename(columns={self.text_col: "text"}, inplace=True)
        kf = KFold(n_splits=n_splits)
        num_labels = len(dataframe[self.label_col].unique())
        folds = []
        for train_index, val_index in kf.split(dataframe):
            train_df = dataframe.iloc[train_index]
            val_df = dataframe.iloc[val_index]
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            folds.append((train_dataset, val_dataset))
        print(
            f"\nDataset loaded. The dataset has {num_labels} labels, {n_splits} splits were made. Each split has {len(train_df)} training items and {len(val_df)} test items.\n{dataframe.head(3)}"
        )
        return folds

    def encode_dataset(self, df, model_name="cmarkea/distilcamembert-base"):
        ds = DatasetDict()
        ds = Dataset.from_pandas(df)

        def modify_features(example):
            example["text"] = example[self.text_col]
            if self.text_col != "text":
                del example[self.text_col]
            return example

        ds = ds.map(modify_features)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize(batch):
            if "camembert" in model_name:
                return tokenizer(
                    batch["text"], padding="max_length", truncation=True, max_length=512
                )
            else:
                return tokenizer(batch["text"], padding=True, truncation=True)

        return ds.map(tokenize, batched=True, batch_size=None)
