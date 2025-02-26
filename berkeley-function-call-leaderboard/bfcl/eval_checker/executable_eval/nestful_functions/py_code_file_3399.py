from typing import Tuple



def is_diagonal(x1: int, y1: int, x2: int, y2: int) -> bool:

    """Checks if the given coordinates are on a diagonal line.



    Args:

        x1: The x-coordinate of the first point.

        y1: The y-coordinate of the first point.

        x2: The x-coordinate of the second point.

        y2: The y-coordinate of the second point.



    Returns:

        True if the coordinates are on a diagonal line, False otherwise.

    """

    assert 0 <= x1 <= 8 and 0 <= y1 <= 8 and 0 <= x2 <= 8 and 0 <= y2 <= 8, "Invalid coordinates"



    return abs(x1 - x2) == abs(y1 - y2)

