from typing import Tuple

from math import sqrt



def point_distance(x1: float, y1: float, x2: float, y2: float) -> float:

    """Computes the distance between two points on a 2D plane using coordinates (x, y).



    Args:

        x1: The x-coordinate of the first point.

        y1: The y-coordinate of the first point.

        x2: The x-coordinate of the second point.

        y2: The y-coordinate of the second point.



    Returns:

        The distance between the two points.

    """

    if y1 == y2:

        if x1 * x2 < 0:

            return abs(x1 - x2)

        else:

            return abs(x1 - x2) + abs(y1 - y2)

    else:

        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

