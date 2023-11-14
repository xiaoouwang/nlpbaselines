# table for the pilot study on french models
import pandas as pd
import torch
import gc
import time

from classifier import Classifier, DatasetLoader
from utils.data import data_sentiment
from utils.log import log_model, add_to_log
from utils.file import fn_datetime, get_current_fn
from utils.report import report_gpu
from utils.file import get_current_fn
from variables import model_list


report_gpu()
filename = "class-one-line"

df = data_sentiment(50, 0.8)
print(df)

# Initialize the classifier
loader = DatasetLoader(text_col="text", label_col="label")

# Load the data
ds_train, ds_val = loader.load_dataset(df)

print("data ok")

m = "cmarkea/distilcamembert-base"
classifier = Classifier(model_name=m, num_labels=2)

epoch = 3
batch_size = 8
learning_rate = 2e-5


classifier.train(
    train_dataset=ds_train,
    validation_dataset=ds_val,
    epochs=epoch,
    batch_size=batch_size,
    learning_rate=learning_rate,
    output_dir=f"./results/",
    seed=43
)

print("F1 Score", classifier.f1)
print("Accuracy Score", classifier.accuracy)
print("Recall Score", classifier.recall)
