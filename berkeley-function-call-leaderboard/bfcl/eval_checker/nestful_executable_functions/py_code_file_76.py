import numpy as np



def create_conv_filter(image_shape: tuple, kernel_size: tuple) -> np.ndarray:

    """Creates a convolutional filter with the specified shape.



    The filter has the same number of channels as the image and is initialized with random values.



    Args:

        image_shape: A tuple of three elements, where the first two elements represent the height and width of an image,

            and the last element represents the number of channels in the image.

        kernel_size: A tuple of two elements, where the first element represents the height of the convolutional kernel,

            and the second element represents the width of the convolutional kernel.



    Returns:

        A 4D array representing the convolutional filter.

    """

    num_channels = image_shape[2]

    filter_height, filter_width = kernel_size

    filter_shape = (filter_height, filter_width, num_channels, num_channels)

    return np.random.random(filter_shape)

