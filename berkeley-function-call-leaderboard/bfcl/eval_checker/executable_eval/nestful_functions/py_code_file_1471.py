import numpy as np



def binary_classification_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    """Computes the binary classification accuracy by comparing the predicted class labels to the true labels.



    Args:

        y_pred: A tensor of size `(m, n)` with the model's predictions for `m` samples, each with `n` class probabilities.

        y_true: A tensor of size `(m, n)` with the true class labels for each sample.



    Returns:

        The binary classification accuracy.

    """

    y_pred_labels = np.argmax(y_pred, axis=1)

    accuracy = np.mean(np.equal(y_pred_labels, y_true))

    return accuracy

