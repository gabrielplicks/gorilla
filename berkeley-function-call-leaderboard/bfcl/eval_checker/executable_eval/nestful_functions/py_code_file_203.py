from typing import List, Tuple

from math import sqrt



def get_closest_point(points: List[Tuple[float, float]], target: Tuple[float, float]) -> Tuple[float, float]:

    """

    Returns the point in the list that is closest to the target point.



    Args:

        points: A list of points.

        target: The target point.



    Raises:

        ValueError: If the list is empty or does not contain the target point.

    """

    if len(points) == 0:

        raise ValueError("List must not be empty")

    if target not in points:

        raise ValueError("Target must be in list")

    distances = {point: sqrt((point[0] - target[0])**2 + (point[1] - target[1])**2) for point in points}

    return min(distances, key=distances.get)

