import numpy as np



def binary_cross_entropy_cost(Y: np.ndarray, Y_hat: np.ndarray) -> float:

    """Calculates the binary cross entropy cost for a binary classification model.



    Args:

        Y: The true labels of the examples.

        Y_hat: The predicted labels of the examples.



    Returns:

        The calculated cost as a single value.

    """

    m = Y.shape[0]  # Number of examples

    cost = (1 / m) * np.sum(-Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat))

    return cost

