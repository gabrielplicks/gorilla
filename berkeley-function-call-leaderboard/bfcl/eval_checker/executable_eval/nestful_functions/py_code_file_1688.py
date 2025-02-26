from typing import Tuple



def find_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:

    """Calculates the distance between two points in 2D space using the Pythagorean theorem.

    Args:

        p1: A tuple of floats representing the x and y coordinates of the first point.

        p2: A tuple of floats representing the x and y coordinates of the second point.

    """

    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

