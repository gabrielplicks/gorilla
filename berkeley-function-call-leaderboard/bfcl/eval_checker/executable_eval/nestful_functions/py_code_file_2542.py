import numpy as np



def leaky_relu(x: np.array, alpha: float) -> np.array:

    """

    Computes the Leaky ReLU activation function for each element in the input array `x`.

    If x is greater than or equal to 0, it returns x. Otherwise, it returns alpha * x.



    Args:

        x: The input array.

        alpha: A positive constant.

    """

    return np.where(x >= 0, x, alpha * x)

