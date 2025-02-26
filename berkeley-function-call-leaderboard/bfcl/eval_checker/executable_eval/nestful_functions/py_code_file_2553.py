import numpy as np



def vector_matrix_multiply(v: np.ndarray, m: np.ndarray) -> np.ndarray:

    """Multiplies a 1-D NumPy array `v` of shape (n, ) with a 2-D NumPy array `m` of shape (n, m).



    Args:

        v: A 1-D NumPy array of shape (n, ).

        m: A 2-D NumPy array of shape (n, m).



    Raises:

        ValueError: If `v` and `m` have incompatible shapes.

    """

    if v.shape[0] != m.shape[0]:

        raise ValueError("v and m must have compatible shapes")

    product = v @ m

    return np.array(product).reshape(m.shape[1], )

