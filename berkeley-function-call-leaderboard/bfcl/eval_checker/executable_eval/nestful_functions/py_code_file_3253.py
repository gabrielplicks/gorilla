import numpy as np



def fliplr(img: np.ndarray) -> np.ndarray:

    """Flips an input image `img` left-to-right.



    Args:

        img: The input image, a Numpy array of shape `(H, W, C)` where `H` is the height, `W` is the width, and `C` is the number of channels.



    Returns:

        A new array with the image flipped horizontally.

    """

    return img[:, ::-1, :]

