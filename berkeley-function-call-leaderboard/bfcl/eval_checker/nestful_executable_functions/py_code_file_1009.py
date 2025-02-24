import numpy as np



def image_padding(image: np.ndarray, padding: int) -> np.ndarray:

    """Pads an image with zeros on the left, top, right, and bottom.



    The padding values are evenly distributed on both sides of the image.



    Args:

        image: A 2D NumPy array of shape (H, W) representing the image.

        padding: The amount of padding to apply to the image.

    """

    height, width = image.shape

    pad_height = (padding, padding)

    pad_width = (padding, padding)

    padded_image = np.pad(image, (pad_height, pad_width), 'constant', constant_values=0)

    return padded_image

