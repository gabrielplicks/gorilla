from typing import List



def count_ways(n: int) -> int:

    """Counts the number of ways to climb a staircase with three different sizes of steps (1, 2, 3).



    Args:

        n: The number of steps in the staircase.



    Returns:

        The number of ways to climb the staircase.

    """

    ways: List[int] = [1, 1, 2]

    for i in range(3, n + 1):

        ways.append(ways[i - 1] + ways[i - 2] + ways[i - 3])

    return ways[n]

