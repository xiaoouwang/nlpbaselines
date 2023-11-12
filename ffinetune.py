from nlpbaselines import utils
import torch
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

def create_ds(df_train, df_val, *args):
    train = Dataset.from_pandas(df_train)
    val = Dataset.from_pandas(df_val)
    if args:
        test = Dataset.from_pandas(args[0])
        ds["test"] = test
    ds = DatasetDict()
    ds["train"] = train
    ds["validation"] = val
    return ds


# Prepare dataset

df_train = pd.read_csv("data/train.txt", sep=";", names=["text", "label", "label_name"])
df_val = pd.read_csv(
    "data/validation.txt", sep=";", names=["text", "label", "label_name"]
)
df_test = pd.read_csv("data/test.txt", sep=";", names=["text", "label", "label_name"])

ds = create_ds(df_train=df_train, df_val=df_val)
ds = ds.class_encode_column("label")
print(ds)

# Prepare metrics for the trainer api


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# encode data

model_name = "distilbert-base-uncased"


def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(batch["text"], padding=True, truncation=True)


batch_size = 64
num_labels = 6

ds_encoded = ds.map(tokenize, batched=True, batch_size=None)

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
seed = 40
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
).to(device)
logging_steps = len(ds_encoded["train"]) // batch_size
output = f"{model_name}-emotion-{batch_size}"
training_args = TrainingArguments(
    output_dir=output,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    # report_to="tensorboard",
    torch_compile=True,  # optimizations
    # optim="adamw_torch_fused", # improved optimizer
    push_to_hub=False,
    seed=seed,
    log_level="error",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ds_encoded["train"],
    eval_dataset=ds_encoded["validation"],
)

trainer.train()
