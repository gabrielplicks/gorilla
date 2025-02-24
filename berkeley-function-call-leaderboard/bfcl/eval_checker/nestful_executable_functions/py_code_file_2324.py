import math



def compute_angle_between(point1: tuple, point2: tuple) -> float:

    """Computes the angle between two points in 2D space in degrees.



    Args:

        point1: A tuple representing a point in 2D space.

        point2: A tuple representing a point in 2D space.



    Returns:

        The angle between the two points in degrees.

    """

    dx = point2[0] - point1[0]

    dy = point2[1] - point1[1]

    return math.degrees(math.atan2(dy, dx))

