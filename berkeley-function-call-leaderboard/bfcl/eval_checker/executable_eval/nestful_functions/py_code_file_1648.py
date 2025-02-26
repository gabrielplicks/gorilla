from typing import Tuple



def is_valid_triangle(a: int, b: int, c: int) -> bool:

    """Checks if the given lengths of the three sides form a valid triangle.



    Args:

        a: The length of the first side.

        b: The length of the second side.

        c: The length of the third side.



    Returns:

        True if the three sides form a valid triangle, False otherwise.

    """

    if not isinstance(a, int) or not isinstance(b, int) or not isinstance(c, int):

        return False

    sides = sorted([a, b, c], reverse=True)

    return sides[0] < sides[1] + sides[2]

