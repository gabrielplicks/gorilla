from typing import Tuple



def calculate_fanout(n_filters: int, kernel_width: int, kernel_height: int) -> int:

    """Calculates the fanout of a convolutional layer in a neural network.

    Args:

        n_filters: The number of filters in the convolutional layer.

        kernel_width: The width of the kernel.

        kernel_height: The height of the kernel.

    """

    assert n_filters > 0 and kernel_width > 0 and kernel_height > 0



    return n_filters * kernel_width * kernel_height

