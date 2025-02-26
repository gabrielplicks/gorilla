import numpy as np

from typing import List, Tuple



def hash_vector(vector: List[float], matrix: np.ndarray) -> int:

    """Computes the hash of a given vector using a given matrix.



    Args:

        vector: The input vector.

        matrix: The matrix to multiply the vector with.

    """

    result = np.dot(vector, matrix)

    result = np.sum(result ** 2)

    return int(result)

