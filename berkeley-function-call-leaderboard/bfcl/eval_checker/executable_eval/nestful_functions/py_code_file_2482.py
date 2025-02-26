from typing import List



def multiple_of_three(n: int) -> List[int]:

    """Returns a list of non-negative integers that are less than or equal to `n` and whose digits sum to a multiple of 3.



    Args:

        n: The maximum value for the integers in the list.



    Returns:

        A list of non-negative integers that are less than or equal to `n` and whose digits sum to a multiple of 3.

    """

    return [i for i in range(n + 1) if sum(divmod(i, 10)) % 3 == 0]

