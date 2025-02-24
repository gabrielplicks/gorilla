from typing import List



def make_range(n: int = 10, start: int = 0) -> List[int]:

    """Converts a range into a list.



    Args:

        n: The end value of the range.

        start: The starting value of the range.



    Returns:

        A list of integers from `start` to `n - 1`.

    """

    return list(range(start, n))

