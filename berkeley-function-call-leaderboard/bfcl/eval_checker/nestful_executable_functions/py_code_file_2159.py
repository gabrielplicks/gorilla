import numpy as np

from typing import List, Tuple



def pairwise_distance(points: List[Tuple[float, float]]) -> np.ndarray:

    """Calculates the distance matrix for a given list of points using vector arithmetic and broadcasting.



    Args:

        points: A list of tuples, each containing the x and y coordinates of a point.



    Returns:

        A matrix of distances, where each element (i, j) represents the distance between point i and point j.

    """

    x = np.array([p[0] for p in points])

    y = np.array([p[1] for p in points])



    x_diff = x.reshape(-1, 1) - x

    y_diff = y.reshape(-1, 1) - y



    distance_matrix = np.sqrt(x_diff ** 2 + y_diff ** 2)

    return distance_matrix

