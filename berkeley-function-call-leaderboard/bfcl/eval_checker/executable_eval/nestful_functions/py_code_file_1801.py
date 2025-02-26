from typing import Tuple



def line_length(x1: float, y1: float, x2: float, y2: float) -> float:

    """Calculates the length of the straight line segment connecting two points.



    Args:

        x1: The x-coordinate of the first point.

        y1: The y-coordinate of the first point.

        x2: The x-coordinate of the second point.

        y2: The y-coordinate of the second point.



    Returns:

        The length of the line segment.

    """

    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

