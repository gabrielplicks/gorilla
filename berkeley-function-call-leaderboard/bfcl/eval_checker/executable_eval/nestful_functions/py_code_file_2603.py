import numpy as np

from typing import List



def euclidean_distance_matrix(X: List[List[float]]) -> np.ndarray:

    """Computes the Euclidean distance matrix of size m x m in O(m x n^2) time complexity.



    Args:

        X: A matrix of size m x n.



    Returns:

        A Euclidean distance matrix of size m x m.

    """

    m = len(X)

    distances = np.zeros((m, m))

    for i in range(m):

        x_i = X[i]

        for j in range(i, m):

            x_j = X[j]

            d_ij = np.sqrt(np.sum(np.square(x_i - x_j)))

            distances[i, j] = distances[j, i] = d_ij

    return distances

