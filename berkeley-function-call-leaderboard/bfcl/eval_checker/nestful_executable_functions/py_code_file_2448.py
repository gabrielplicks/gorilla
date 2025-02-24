from typing import Tuple



def compute_distance_to_line(x: float, m: float, b: float) -> float:

    """Computes the distance of a point (x, y) to a line represented by the function f(x) = m * x + b.



    Args:

        x: The x-coordinate of the point.

        m: The slope of the line.

        b: The y-intercept of the line.



    Returns:

        The Euclidean distance to the line.

    """

    assert x > 0, "x must be a positive number greater than 0"

    assert m > 0, "m must be a positive number"

    assert isinstance(b, (int, float)), "b must be a number"



    y = m * x + b

    x1, y1 = x, 0

    x2, y2 = 0, y

    return abs((y2 - y1) / (x2 - x1))

