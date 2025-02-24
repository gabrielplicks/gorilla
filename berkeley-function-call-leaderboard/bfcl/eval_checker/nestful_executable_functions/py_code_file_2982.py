import numpy as np



def convert_to_ratio_array(arr: np.ndarray) -> np.ndarray:

    """Converts the input array to a numpy array if it is not already, checks if all elements are zero, calculates the sum of all elements, and then divides each element by the sum. Finally, it rounds each element to 4 decimal places.



    Args:

        arr: The input numpy array.



    Returns:

        A numpy array with the same dimensions as the input array, where each element is the ratio of the corresponding element in the input array to the sum of all elements in the input array, rounded to 4 decimal places.

    """

    if not isinstance(arr, np.ndarray):

        arr = np.array(arr)

    if (arr == 0).all():

        return np.zeros_like(arr)

    return np.round(arr / arr.sum(), 4)

