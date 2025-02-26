import numpy as np



def compute_mean_of_rows(array_2d: np.ndarray) -> np.ndarray:

    """Computes the mean of each row in a 2D array with a given shape.



    Args:

        array_2d: The input 2D array.



    Returns:

        A 1D array of the same length, where each element is the average value of the corresponding row in the input array.



    Raises:

        ValueError: If the input array has fewer than 2 dimensions.

    """

    rows, columns = array_2d.shape

    if len(array_2d.shape) < 2:

        raise ValueError("Input array must be 2D")

    mean_vals = []

    for i in range(rows):

        row = array_2d[i]

        mean = sum(row) / len(row)

        mean_vals.append(mean)



    return np.array(mean_vals)

