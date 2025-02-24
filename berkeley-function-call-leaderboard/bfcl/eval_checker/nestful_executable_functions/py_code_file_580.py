import numpy as np



def convert_numpy_array_to_list(numpy_arr: np.ndarray) -> list:

    """Converts a 1D NumPy array to a list.



    Args:

        numpy_arr: The 1D NumPy array to be converted.



    Returns:

        A list containing the elements of the input NumPy array.

    """

    return numpy_arr.tolist()

