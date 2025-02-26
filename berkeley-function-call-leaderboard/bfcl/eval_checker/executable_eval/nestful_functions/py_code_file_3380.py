import numpy as np



def scale_image_data(image_data: np.ndarray, factor: int) -> np.ndarray:

    """Scales an input image data array by a specified factor.



    Args:

        image_data: The input image data array.

        factor: The scaling factor.

    """

    assert isinstance(factor, int) and factor > 0, "Factor must be a positive integer"



    scaled_image = np.multiply(image_data, factor)

    return scaled_image

