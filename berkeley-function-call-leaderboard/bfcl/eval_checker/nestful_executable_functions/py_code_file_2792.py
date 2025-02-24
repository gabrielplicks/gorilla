import numpy as np



def find_max_indices(data: np.ndarray) -> tuple:

    """Finds the indices of the maximum value in every dimension of a 3D array.



    Args:

        data: A 3D array of size (M, N, O), where M is the number of y-coordinates,

              N is the number of x-coordinates, and O is the number of z-coordinates.

              The three dimensions correspond to the three coordinates.



    Returns:

        A tuple of indices of the maximum value in every dimension of `data`.

    """

    max_value = np.max(data)

    max_index = np.argmax(data)

    max_indices = np.unravel_index(max_index, data.shape)



    return max_indices

