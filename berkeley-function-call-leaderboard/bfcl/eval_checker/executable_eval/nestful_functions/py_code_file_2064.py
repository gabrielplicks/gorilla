from typing import Tuple



def calculate_x(m: float, c: float) -> float:

    """Calculates the value of x for a linear equation given the slope and intercept.

    The line equation is represented by `y = mx + c`, where m is the slope and c is the intercept.

    The function takes two arguments: `m` for the slope and `c` for the intercept.

    Args:

        m: The slope of the line.

        c: The intercept of the line.

    Returns:

        The value of x.

    """

    return (0 - c) / m

