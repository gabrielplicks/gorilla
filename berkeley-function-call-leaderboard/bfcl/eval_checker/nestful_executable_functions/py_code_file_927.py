import numpy as np



def vector_to_matrix(vector: List[int]) -> np.ndarray:

    """Transforms a vector of size N into a matrix of size N × N.

    Each element of the matrix should be an integer value, which is the index of the corresponding element in the vector.

    Uses the NumPy library for matrix manipulation.

    Args:

        vector: A list of integers representing the vector.

    Returns:

        A 2D NumPy array representing the matrix.

    """

    vector_array = np.array(vector)

    matrix = np.zeros((len(vector), len(vector)), dtype=int)

    for i in range(len(vector_array)):

        matrix[i, i] = vector_array[i]



    return matrix

