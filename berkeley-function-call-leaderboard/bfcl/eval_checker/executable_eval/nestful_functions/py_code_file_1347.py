from typing import List



def count_up(n: int) -> List[int]:

    """

    Returns a list of `n` integers in increasing order beginning with 0.

    If `n` is 0, then the function returns an empty list.

    If `n` is a negative integer, then the function raises a custom `ValueError` with a message that specifies the problem.



    Args:

        n: The number of integers to generate.



    Raises:

        ValueError: If `n` is a negative integer.

    """

    if n < 0:

        raise ValueError("n cannot be negative.")

    return list(range(n))

