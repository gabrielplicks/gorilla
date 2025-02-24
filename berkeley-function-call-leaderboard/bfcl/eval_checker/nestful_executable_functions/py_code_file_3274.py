from typing import List, Optional



def generate_range(n: int, start: Optional[int] = None) -> List[int]:

    """

    Returns a list of integers in the range between 0 and the parameter `n`, where the step value is 1.

    The function accepts an optional `start` parameter that sets the starting number of the range.

    If `start` is not provided, return a range from 0 to `n`.



    Args:

        n: The end of the range.

        start: The start of the range. Default is 0.



    Returns:

        A list of integers in the range between 0 and `n`.

    """

    start = start or 0

    return list(range(start, n + 1))

