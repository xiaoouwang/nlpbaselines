import pandas as pd


def extract_best_per_run(lst, epoch=3):
    """
    Extracts the best (maximum) value from each run of n epochs.

    Args:
    lst (list): A list of values.
    epoch (int): The number of epochs in each run. Default is 3.

    Returns:
    list: A list of the best (maximum) value from each run.
    """

    # Split the list into sub-lists of `epoch` elements each
    runs = [lst[i : i + epoch] for i in range(0, len(lst), epoch)]

    # Find the best (maximum) value in each run
    best_per_run = [max(run) for run in runs]

    return best_per_run


# read the buggy_cv.csv file
df = pd.read_csv("cv_results.csv")

# apply extract_best_per_run() function on each row of f1, recall and accuracy columns

import ast


# Apply the function to each metric column
for column in ["f1", "accuracy", "recall"]:
    df[column] = df[column].apply(
        ast.literal_eval
    )  # Convert string representation of list to actual list
    df[f"best_{column}"] = df[column].apply(lambda x: extract_best_per_run(x))
