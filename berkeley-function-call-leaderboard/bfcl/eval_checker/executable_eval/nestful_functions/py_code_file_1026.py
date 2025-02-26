import numpy as np



def l2_regularizer(W: np.ndarray) -> float:

    """Calculates L2 weight regularization loss for a given weight matrix.

    Args:

        W: The weight matrix.

    """

    norm = np.sqrt(np.sum(np.square(W)))

    loss = norm ** 2

    return loss

