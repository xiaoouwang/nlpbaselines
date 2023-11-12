from nlpbaselines import utils
import torch
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score


# train_ds, val_ds, test_ds = load_dataset(
#     'allocine',
#     split=['train', 'validation', 'test']
# )
# train_ds.to_csv("data/train-fr.txt")
# val_ds.to_csv("data/validation-fr.txt")
# test_ds.to_csv("data/test-fr.txt")

