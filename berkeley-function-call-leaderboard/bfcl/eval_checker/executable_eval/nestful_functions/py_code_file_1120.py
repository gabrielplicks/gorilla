from typing import Union



def max_dist_from_x_axis(m: Union[float, str], b: float) -> float:

    """Calculates the maximum distance of any point on the line from the x-axis.

    Args:

        m: The slope of the line.

        b: The y-intercept of the line.

    """

    if m == float('inf'):

        return abs(b)

    else:

        x = -b / m

        return abs(x)

