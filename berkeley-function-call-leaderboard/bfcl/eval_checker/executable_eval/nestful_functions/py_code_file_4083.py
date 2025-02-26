import math

from typing import List, Tuple



def distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:

    """Calculates the distance between two points in a 2D space.



    Args:

        point_a: The first point.

        point_b: The second point.

    """

    x1, y1 = point_a

    x2, y2 = point_b

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



def calculate_distance(points: List[Tuple[float, float]]) -> float:

    """Calculates the distance between the first point and the second point in a list of points.



    Args:

        points: A list of points.

    """

    point_a, point_b = points[:2]

    if len(points) > 2:

        point_a = points[0]

    return distance(point_a, point_b)

