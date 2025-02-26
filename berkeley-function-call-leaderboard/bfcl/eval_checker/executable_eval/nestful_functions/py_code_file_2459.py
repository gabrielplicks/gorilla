from typing import Tuple



def point_on_line(a: float, b: float, c: float, x0: float, y0: float) -> int:

    """Determines whether a point lies on/above/below a line.



    Args:

        a: The coefficient of x in the line equation.

        b: The coefficient of y in the line equation.

        c: The constant term in the line equation.

        x0: The x-coordinate of the point.

        y0: The y-coordinate of the point.



    Returns:

        0 if the point lies on the line, 1 if it is above the line, and -1 if it is below the line.

    """

    value = a * x0 + b * y0 + c

    if value == 0:

        return 0

    elif value > 0:

        return 1

    else:

        return -1

