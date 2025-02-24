from typing import List



def distribute_balls(m: int, n: int) -> List[List[int]]:

    """Distribute n balls into m bins. Each bin can hold arbitrarily many balls.



    Args:

        m: The number of bins.

        n: The number of balls.



    Returns:

        A list of n-element lists, where each list represents a bin and its elements are the indices of the balls in that bin.

    """

    assert m > 0 and n > 0, "Both m and n must be positive integers"

    return [[i for i in range(n) if i % m == j] for j in range(m)]

