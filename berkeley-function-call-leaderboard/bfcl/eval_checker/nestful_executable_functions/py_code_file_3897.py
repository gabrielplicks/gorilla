from typing import List



def rolling_sum(n: int, k: int) -> List[int]:

    """Creates a new list with each element being the sum of the previous `k` elements.



    Args:

        n: The size of the list.

        k: The number of previous elements to sum.

    """

    return [sum(range(i, max(0, i - k), -1)) for i in range(n)]

