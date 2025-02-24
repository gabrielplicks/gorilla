import numpy as np



def stable_sigmoid(x: np.ndarray) -> np.ndarray:

    """Calculates the sigmoid of a vector or scalar using a numerically stable implementation.



    Args:

        x: The input vector or scalar.

    """

    exp_x = np.exp(-x)

    return 1.0 / (1.0 + exp_x)

