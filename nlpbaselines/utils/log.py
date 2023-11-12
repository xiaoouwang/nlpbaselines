import pandas as pd

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
            "model_param",
            "model_size",
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
    model_param,
    model_size,
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
        model_param,
        model_size,
    ]
    return log

