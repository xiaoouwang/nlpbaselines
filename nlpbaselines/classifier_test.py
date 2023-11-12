# table for the pilot study on french models
import pandas as pd
import torch
import random
# from nlpbaselines.classifier import Classifier, DatasetLoader

import gc
import time


def get_current_fn(ext):
    import os

    filename = os.path.basename(__file__)
    return filename[: -(len(ext) + 1)]


def log_model():
    log = pd.DataFrame(
        columns=[
            "model_name",
            "f1",
            "accuracy",
            "recall",
            "training_time",
            "data_size",
            "batch_size",
            "learning_rate",
            "epochs",
        ]
    )
    return log


# create a function to add data to the log
def add_to_log(
    log,
    model_name,
    f1,
    accuracy,
    recall,
    training_time,
    data_size,
    batch_size,
    learning_rate,
    epochs,
):
    log.loc[len(log) + 1] = [
        model_name,
        f1,
        accuracy,
        recall,
        training_time,
        data_size,
        batch_size,
        learning_rate,
        epochs,
    ]
    return log


# Toy data
data = {
    "text": [f"This is sentence {i}" for i in range(30)],
    "label": [random.randint(0, 1) for _ in range(30)],
    "split": ["train" if i < 25 else "validation" for i in range(30)],
}
df = pd.DataFrame(data)

# Initialize the classifier
loader = DatasetLoader(text_col="text", label_col="label")
ds_train, ds_validation = loader.load_dataset(df)

print("data ok")

# create a pandas dataframe to log model name, f1, accuracy, recall, training time, data_size, batch size, learning rate, epochs

log = log_model()


# check the number of available gpus
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# model_lists = [
#     "bert-base-uncased",
#     "distilbert-base-uncased",
#     "bert-base-multilingual-uncased",
#     "camembert-base",
#     "camembert/camembert-large",
#     "camembert/camembert-base-ccnet",
#     "camembert/camembert-base-ccnet-4gb",
#     "camembert/camembert-base-oscar-4gb",
#     "camembert/camembert-base-wikipedia-4gb",
#     "flaubert/flaubert_small_cased",
#     "flaubert/flaubert_base_uncased",
#     "flaubert/flaubert_base_cased",
#     "flaubert/flaubert_large_cased",
# ]

model_lists = ["distilbert-base-uncased"]
filename = get_current_fn("py")
from datetime import datetime

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

for m in model_lists:
    print(f"model {m} start")
    epoch = 2
    batch_size = 10
    learning_rate = 2e-5
    classifier = Classifier(model_name=m, num_labels=2, use_multi_gpu=False)
    start_time = time.time()
    classifier.train(
        ds_train,
        ds_validation,
        epochs=epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=f"./results/{filename}-{formatted_datetime}/{m}/",
    )
    print("F1 Score:", classifier.f1)
    print("Accuracy Score:", classifier.accuracy)
    print("Recall Score:", classifier.recall)

    end_time = time.time()
    training_time = f"{end_time - start_time:.2f}"
    log = add_to_log(
        log,
        m,
        classifier.f1,
        classifier.accuracy,
        classifier.recall,
        training_time,
        len(df),
        10,
        2e-5,
        epoch,
    )
    del classifier
    gc.collect()
    torch.cuda.empty_cache()
    # for slurm
    print(f"model {m} done, results saved to {filename}.csv")
    log.to_csv(f"{filename}.csv", index=False)
