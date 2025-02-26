import numpy as np



def diag(x: np.ndarray) -> np.ndarray:

    """Constructs a new tensor with the same value as `x` but in a diagonal layout.



    Args:

        x: A square NumPy array in float32 format.



    Raises:

        ValueError: If `x` is not a NumPy array in float32 format or if `x` is not square.

    """

    if not isinstance(x, np.ndarray) or x.dtype != np.float32:

        raise ValueError("Input must be a NumPy array in float32 format.")



    if x.shape[0] != x.shape[1]:

        raise ValueError("Input must be a square array.")



    output = np.zeros_like(x)

    output[np.diag_indices_from(output)] = x[np.diag_indices_from(x)]

    return output

