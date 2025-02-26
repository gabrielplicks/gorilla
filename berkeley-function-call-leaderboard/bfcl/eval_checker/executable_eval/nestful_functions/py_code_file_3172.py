import numpy as np



def is_binary_image(image: np.ndarray) -> bool:

    """Checks if the input image is a binary image, i.e., only contains two distinct values.



    Args:

        image: A numpy array representing an image.



    Returns:

        A boolean value indicating whether the image is binary.

    """

    binary_values = (image == 0) | (image == 1)

    return not binary_values.any() or binary_values.all()

