from typing import Tuple



def check_precedence(a1: float, b1: float, a2: float, b2: float) -> bool:

    """Checks if two intervals overlap.



    Args:

        a1: The start of the first interval.

        b1: The end of the first interval.

        a2: The start of the second interval.

        b2: The end of the second interval.



    Returns:

        True if the intervals overlap, False otherwise.

    """

    if a1 <= a2 <= b1 or a2 <= a1 <= b2:

        return True

    return False

