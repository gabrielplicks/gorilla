import numpy as np



def combine_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:

    """Combines three 2D matrices (R, G, B channels) into a new 2D matrix P.



    Each pixel in the new matrix is calculated as:

    P = R^2 + G^2 + B^2



    Args:

        r: The red channel as a 2D matrix.

        g: The green channel as a 2D matrix.

        b: The blue channel as a 2D matrix.



    Returns:

        The new 2D matrix P.

    """

    r = np.array(r)

    g = np.array(g)

    b = np.array(b)

    p = r**2 + g**2 + b**2



    return p

