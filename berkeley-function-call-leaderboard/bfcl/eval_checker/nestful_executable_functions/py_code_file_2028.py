import numpy as np



def extract_red_green_channels(image: np.ndarray) -> tuple:

    """

    Extracts the red and green channels from an RGB image represented as a NumPy array.

    Args:

        image: A NumPy array representing an RGB image.

    Returns:

        A tuple containing two NumPy arrays representing the red and green channels respectively.

    """

    red_channel = image[:, :, 0]

    green_channel = image[:, :, 1]



    return red_channel, green_channel

