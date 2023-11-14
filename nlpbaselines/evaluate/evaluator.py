from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from metrics import compute_metrics
from nlpbaselines.utils.gpu import select_device
from nlpbaselines.utils.data import DatasetLoader
import numpy as np
import pandas as pd
import torch


class Evaluator:
    def __init__(self, df, model_name, num_labels=2, text_col="text", lab_col="label"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.df = df
        self.df_loss = None
        self.text_col = text_col
        self.lab_col = lab_col
        self.device = select_device()
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels
        ).to(self.device)
        self.trainer = Trainer(model=self.model, compute_metrics=compute_metrics)
        self.ds_encoded = self.encode(self.df)
        self.preds = self.trainer.predict(self.ds_encoded)
        self.predicted_labels = np.argmax(self.preds.predictions, axis=1).tolist()
        self.df["predicted_label"] = self.predicted_labels
        self.df_encoded_features = None

    def encode(self, df):
        loader = DatasetLoader(text_col=self.text_col, label_col=self.lab_col)
        ds_encoded = loader.encode_dataset(df, model_name=self.model_name)
        return ds_encoded

    def compute_loss(self):
        # Look for biggest loss
        from torch.nn.functional import cross_entropy

        # Compute loss values
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def forward_pass_with_label(batch):
            # Place all input tensors on the same device as the model
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in tokenizer.model_input_names
            }

            with torch.no_grad():
                output = self.model(**inputs)
                pred_label = torch.argmax(output.logits, axis=-1)
                loss = cross_entropy(
                    output.logits, batch["label"].to(device), reduction="none"
                )

                # Place outputs on CPU for compatibility with other dataset columns
                return {
                    "loss": loss.cpu().numpy(),
                    "predicted_label": pred_label.cpu().numpy(),
                }

        self.ds_encoded.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )
        # Compute loss values
        df_loss = self.ds_encoded.map(
            forward_pass_with_label, batched=True, batch_size=16
        )
        df_loss.set_format("pandas")
        df_loss = df_loss[:][:]
        print("after")
        print(df_loss)
        self.ds_encoded.set_format("pandas")
        self.df_loss = df_loss

    def compute_representation(self):
        from transformers import AutoModel

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)

        def extract_hidden_states(batch):
            # Place model inputs on the GPU
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in tokenizer.model_input_names
            }
            # Extract last hidden states
            with torch.no_grad():
                last_hidden_state = model(**inputs).last_hidden_state
            # Return vector for [CLS] token
            return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

        self.ds_encoded.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        ds_encoded_features = self.ds_encoded.map(extract_hidden_states, batched=True)
        ds_encoded_features.set_format("pandas")
        ds_encoded_features = ds_encoded_features[:][:]
        self.df_encoded_features = ds_encoded_features


if __name__ == "__main__":
    device = select_device()
    model_name = "../models/wang-distilcamembert"
    df = pd.read_csv("../tutos/data/test-fr-sampled.txt", sep=",")
    Evaluator = Evaluator(df, model_name)
    Evaluator.compute_loss()
    df_loss = Evaluator.df_loss

    # highest losses
    def label_int2str(row):
        return {0: "negative", 1: "positive"}[row]

    print(df_loss)
    print(type(df_loss))
    df_loss["label"] = df_loss["label"].apply(label_int2str)
    df_loss["predicted_label"] = df_loss["predicted_label"].apply(label_int2str)
    for i, row in df_loss.sort_values("loss", ascending=False).head(10).iterrows():
        print(
            f"{row['text'][:400]}\nLabel: {row['label']}\nPredicted label: {row['predicted_label']}\n"
        )

    from nlpbaselines.visualize.plots import plot_confusion_matrix

    plot_confusion_matrix(
        Evaluator.df["predicted_label"].tolist(),
        Evaluator.df["label"].tolist(),
        ["negative", "positive"],
    )

    Evaluator.compute_representation()
    print(Evaluator.df_encoded_features)
    X = np.array(Evaluator.df_encoded_features["hidden_state"].tolist())
    y = np.array(Evaluator.df_encoded_features["label"].tolist())
    from umap import UMAP
    from sklearn.preprocessing import MinMaxScaler

    # Scale features to [0,1] range
    X_scaled = MinMaxScaler().fit_transform(X)
    import matplotlib.pyplot as plt

    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    # Create a DataFrame of 2D embeddings
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    axes = axes.flatten()
    cmaps = ["Greys", "Greens"]
    labels = ["Negative", "Positive"]

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(
            df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,)
        )
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    # TODO add if true save just as the confusion matrix plot
    plt.show()
