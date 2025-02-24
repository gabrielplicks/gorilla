import numpy as np



def raw_moment(img: np.ndarray, p: int, q: int) -> float:

    """Computes the raw moment of an image with a given order.



    Args:

        img: The input image.

        p: The horizontal order.

        q: The vertical order.



    Returns:

        The computed raw moment.

    """

    return np.sum(img**p * img**q)

