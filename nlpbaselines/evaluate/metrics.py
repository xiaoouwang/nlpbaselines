from typing import Dict, Any
from sklearn.metrics import accuracy_score, recall_score, f1_score

def compute_metrics(pred) -> Dict[str, Any]:
    """
    Computes the accuracy, recall, and F1 score for a given set of predictions.

    Args:
        pred: A `transformers.trainer_utils.EvalPrediction` object containing the predicted labels and logits.

    Returns:
        A dictionary containing the accuracy, recall, and F1 score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average="weighted")
    f1 = float("{:.2f}".format(f1))
    accuracy = float("{:.2f}".format(accuracy))
    recall = float("{:.2f}".format(recall))
    return {"accuracy": accuracy, "recall": recall, "f1": f1}
