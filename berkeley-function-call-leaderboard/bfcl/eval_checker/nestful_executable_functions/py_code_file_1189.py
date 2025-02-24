import numpy as np



def logistic(x: np.ndarray) -> np.ndarray:

    """Calculates the logistic function of x.

    Args:

        x: The input array.

    """

    return 1 / (1 + np.exp(-x))



def logistic_derivative(x: np.ndarray) -> np.ndarray:

    """Calculates the derivative of the logistic function of x.

    Args:

        x: The input array.

    """

    return logistic(x) * (1 - logistic(x))

