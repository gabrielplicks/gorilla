from typing import Tuple



def convert_to_range_value(x: int, y: int) -> int:

    """Converts two integers representing a point in the domain to a single integer that represents the corresponding range value.



    Args:

        x: The x-coordinate of the point in the domain.

        y: The y-coordinate of the point in the domain.



    Returns:

        The equivalent range value computed using the formula `y = 20*x + 20`.

    """

    range_value = 20 * x + y

    return range_value

