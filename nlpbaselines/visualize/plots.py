import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nlpbaselines.utils.file import fn_datetime


def plot_confusion_matrix(y_preds, y_true, labels, save=True):
    """
    Plots a normalized confusion matrix given predicted and true labels.

    Args:
    - y_preds (list): List of predicted labels.
    - y_true (list): List of true labels.
    - labels (list): List of label names.

    Returns:
    - None
    """
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    if save:
        plt.savefig(fn_datetime() + "-confusion-matrix.png")
    plt.show()
    