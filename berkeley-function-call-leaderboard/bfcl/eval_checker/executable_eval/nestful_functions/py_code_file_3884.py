import numpy as np

from typing import List, Tuple



def sparse_coordinate_representation(array: np.ndarray) -> Tuple[List[int], List[int], List[int]]:

    """Converts a dense numpy array to a sparse array representation using its coordinates.



    Args:

        array: The input dense numpy array.



    Returns:

        A tuple of three lists: `x`, `y`, and `v`. Each element in `x` and `y` is the x and y coordinate of the

        corresponding value in the `v` list.

    """

    x, y, v = [], [], []

    for i, row in enumerate(array):

        for j, value in enumerate(row):

            if value != 0:

                x.append(i)

                y.append(j)

                v.append(value)

    return x, y, v

