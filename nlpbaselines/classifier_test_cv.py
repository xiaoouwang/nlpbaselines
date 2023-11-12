# table for the pilot study on french models
import pandas as pd
import torch
import gc
import time

from nlpbaselines.classifier import Classifier, DatasetLoader
from nlpbaselines.utils.data import data_sentiment
from nlpbaselines.utils.log import log_model, add_to_log
from nlpbaselines.utils.file import fn_datetime, get_current_fn
from nlpbaselines.utils.report import report_gpu

# from nlpbaselines.utils.file import get_current_fn
from nlpbaselines.variables import model_list


report_gpu()

df = data_sentiment(50, 0.8)
print(df)
# prepare data
loader = DatasetLoader(text_col="text", label_col="label")
folds = loader.load_dataset_cv(df, n_splits=5)
print("data ok")

# create a pandas dataframe to log model name, f1, accuracy, recall, training time, data_size, batch size, learning rate, epochs, model param, model size
log = log_model()

# model_list = ["distilbert-base-uncased"]
filename = get_current_fn(__file__, "py")
print(filename)
formatted_datetime = fn_datetime()

model_list = ["distilbert-base-uncased"]

for m in model_list:
    print(f"model {m} start")
    epoch = 2
    batch_size = 10
    learning_rate = 2e-5
    classifier = Classifier(model_name=m, num_labels=2, use_multi_gpu=False)
    start_time = time.time()
    classifier.train_cv(
        folds,
        epochs=epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=f"./results/{filename}-{formatted_datetime}/{m}/",
    )
    print("F1 Scores for each fold:", classifier.all_f1_scores)
    print("Accuracy Scores for each fold:", classifier.all_accuracy_scores)
    print("Recall Scores for each fold:", classifier.all_recall_scores)

    end_time = time.time()
    training_time = f"{end_time - start_time:.2f}"
    log = add_to_log(
        log,
        m,
        classifier.all_f1_scores,
        classifier.all_accuracy_scores,
        classifier.all_recall_scores,
        training_time,
        len(df),
        10,
        2e-5,
        epoch,
        classifier.model_param,
        classifier.model_size,
    )
    del classifier
    gc.collect()
    torch.cuda.empty_cache()
    # for slurm
    print(f"model {m} done, results saved to {filename}.csv")
    log.to_csv(f"{filename}.csv", index=False)
