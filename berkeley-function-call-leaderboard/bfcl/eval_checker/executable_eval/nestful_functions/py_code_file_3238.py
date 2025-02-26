from typing import List, Tuple

from math import sqrt



def maximum_distance(points: List[Tuple[float, float]]) -> float:

    """Finds the maximum distance between any two points in a given list of points in 2D space.



    Args:

        points: A list of points in 2D space, where each point is represented as a tuple of (x, y) coordinates.



    Returns:

        The maximum distance between any two points in the given list.

    """

    max_distance = 0

    for i in range(len(points)):

        for j in range(i + 1, len(points)):

            dx = points[i][0] - points[j][0]

            dy = points[i][1] - points[j][1]

            distance = sqrt(dx**2 + dy**2)

            if distance > max_distance:

                max_distance = distance

    return max_distance

