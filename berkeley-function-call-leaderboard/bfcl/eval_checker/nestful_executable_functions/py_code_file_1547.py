import numpy as np



def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    """Performs element-wise multiplication on two NumPy arrays.



    Args:

        a: The first NumPy array.

        b: The second NumPy array.



    Raises:

        TypeError: If either input is not a NumPy array.

    """

    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):

        raise TypeError("Both inputs must be NumPy arrays.")

    return a * b

