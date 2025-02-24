import numpy as np



def compute_vector_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:

    """Calculates the Euclidean distance between two 1-dimensional vectors.



    Args:

        vector1: The first 1-dimensional vector.

        vector2: The second 1-dimensional vector.



    Returns:

        The Euclidean distance between the two vectors.

    """

    difference_vector = np.subtract(vector1, vector2)

    squared_difference_vector = np.square(difference_vector)

    sum_of_squared_differences = np.sum(squared_difference_vector)

    distance = np.sqrt(sum_of_squared_differences)

    return distance

