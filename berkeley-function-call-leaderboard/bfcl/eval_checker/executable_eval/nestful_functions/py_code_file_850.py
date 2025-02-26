import numpy as np



def broadcast_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    """Adds two NumPy arrays `a` and `b` of compatible shapes by performing broadcasting on the arrays to match each other's shapes.



    Args:

        a: The first NumPy array.

        b: The second NumPy array.



    Raises:

        ValueError: If the arrays have incompatible shapes.

    """

    if a.shape == b.shape:

        return a + b



    if a.shape == (1,) and b.shape != (1,) and len(b.shape) > 0:

        return a[0] + b



    if b.shape == (1,) and a.shape != (1,) and len(a.shape) > 0:

        return a + b[0]



    raise ValueError("Arrays have incompatible shapes.")

