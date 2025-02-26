import numpy as np



def pad_with_max_value(input_array: np.ndarray) -> np.ndarray:

    """Pads an input array with the maximum value along the H dimension.

    The padded array has the same width as the input array and a height that is a power of 2.

    Args:

        input_array: The input array of size (H, W).

    Returns:

        The padded array.

    """

    max_value = np.max(input_array, axis=0)

    height = input_array.shape[0]

    next_power_of_2 = 2 ** (height.bit_length())

    padded_array = np.full((next_power_of_2, input_array.shape[1]), max_value)

    padded_array[:input_array.shape[0], :] = input_array



    return padded_array

