import random
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import torch
import gc
import time
from torch.nn import DataParallel
import torch
import os
from sklearn.model_selection import KFold
from datetime import datetime



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


class Classifier:
    def __init__(
        self,
        model_name="camembert/camembert-base-ccnet-4gb",
        num_labels=2,
        use_multi_gpu=False,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.f1 = None
        self.accuracy = None
        self.recall = None
        self.use_multi_gpu = use_multi_gpu
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.model_param = self.get_model_size(self.model)[0]
        self.model_size = self.get_model_size(self.model)[1]
        self.model = None
        print(
            f"\nModel loaded. We will finetune {self.model_name} with {self.num_labels} labels."
        )
        self.all_f1_scores = []
        self.all_accuracy_scores = []
        self.all_recall_scores = []

        # Additional configurations can be added here

    def get_model_size(self, model):
        # from https://camembert-model.fr/posts/tutorial/
        param_size = 0
        param_count = 0
        for param in model.parameters():
            param_count += param.nelement()
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return param_count, f"{size_all_mb:.2f}MB"

    def tokenize_function(self, examples):
        if "camembert" in self.model_name:
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )
        else:
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True
            )

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average="weighted")
        f1 = f1_score(labels, preds, average="weighted")
        accuracy = float("{:.2f}".format(accuracy))
        recall = float("{:.2f}".format(recall))
        f1 = float("{:.2f}".format(f1))
        self.f1 = f1
        self.accuracy = accuracy
        self.recall = recall
        self.all_f1_scores.append(f1)
        self.all_accuracy_scores.append(accuracy)
        self.all_recall_scores.append(recall)
        return {"accuracy": accuracy, "recall": recall, "f1": f1}

    def train(
        self,
        train_dataset,
        validation_dataset,
        epochs=2,
        batch_size=10,
        learning_rate=2e-5,
        output_dir="./results/",
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = DataParallel(self.model)
        model.to(self.device)
        dstrain_encoded = train_dataset.map(
            self.tokenize_function, batched=True, batch_size=None
        )
        dsvalidation_encoded = validation_dataset.map(
            self.tokenize_function, batched=True, batch_size=None
        )
        print("data encoded")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            seed=42,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy = "no"
        )

        # training_args.set_save(strategy="epoch")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dstrain_encoded,
            eval_dataset=dsvalidation_encoded,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(output_dir)
        del trainer, model
        # free gpu memory
        gc.collect()
        torch.cuda.empty_cache()

    def train_cv(self, folds, epochs=2, batch_size=10, learning_rate=2e-5, output_dir="./results/"):
        for i, (train_dataset, validation_dataset) in enumerate(folds):
            print(f"Training on fold {i+1}/{len(folds)}")
            self.train(
                train_dataset, validation_dataset, epochs, batch_size, learning_rate, output_dir+f"fold-{i+1}/"
            )
