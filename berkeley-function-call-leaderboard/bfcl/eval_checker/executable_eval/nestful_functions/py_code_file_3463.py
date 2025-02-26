from typing import Tuple



def count_increments(start: int, end: int, increment: int) -> int:

    """Calculates the number of increments in the range from start to end inclusive, with the provided increment.



    Args:

        start: The starting value of the range.

        end: The ending value of the range.

        increment: The increment value.



    Returns:

        The number of increments in the range.

    """

    return len(range(start, end + 1, increment))

