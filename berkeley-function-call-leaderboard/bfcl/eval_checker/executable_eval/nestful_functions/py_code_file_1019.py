import numpy as np

from itertools import chain

from typing import List, Union



def convert_matrix_to_vector(A: List[List[Union[int, float]]], column: bool) -> np.ndarray:

    """Converts a matrix to a vector by concatenating its columns or rows.

    Args:

        A: The input matrix.

        column: If True, concatenate columns. If False, concatenate rows.

    Raises:

        ValueError: If column is neither True nor False.

    """

    if column:

        return np.array(list(chain.from_iterable(A)))

    else:

        return np.array(list(chain.from_iterable(zip(*A))))

