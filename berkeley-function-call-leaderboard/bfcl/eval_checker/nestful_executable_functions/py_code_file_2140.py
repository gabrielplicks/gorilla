from typing import Tuple



def point_in_rect(point: Tuple[float, float], rect: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:

    """Checks whether a point lies strictly within a rectangle, excluding the boundary.



    Args:

        point: A tuple `(x, y)` representing the point.

        rect: A tuple of two tuples `((xmin, ymin), (xmax, ymax))` representing the rectangle.



    Returns:

        True if the point lies strictly within the rectangle, False otherwise.

    """

    (x, y), ((xmin, ymin), (xmax, ymax)) = point, rect

    return xmin < x < xmax and ymin < y < ymax

