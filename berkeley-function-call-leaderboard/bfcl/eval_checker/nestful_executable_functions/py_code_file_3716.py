from typing import List



def find_negative(x: List[int]) -> int:

    """Finds the index of the first negative value in a list of integers `x`.



    Args:

        x: A list of integers.



    Returns:

        The index of the first negative value in `x` if it exists, otherwise -1.

    """

    for i, val in enumerate(x):

        if val < 0:

            return i

    return -1

