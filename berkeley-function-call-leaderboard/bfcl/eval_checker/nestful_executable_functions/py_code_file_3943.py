from typing import Tuple

from math import sqrt



def dist_to_point(point: Tuple[float, float], center: Tuple[float, float]) -> float:

    """Calculates the distance between a point and a center point on a two-dimensional plane.



    Args:

        point: A tuple (x, y) representing a point on the plane.

        center: A tuple (x, y) representing the center point.



    Returns:

        The distance as a floating-point number.

    """

    x1, y1 = point

    x2, y2 = center

    dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return dist

