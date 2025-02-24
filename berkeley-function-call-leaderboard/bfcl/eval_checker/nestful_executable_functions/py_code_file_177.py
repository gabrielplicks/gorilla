from typing import List



def arithmetic_list(n: int) -> List[int]:

    """Creates a list of length n containing the values 1, 2, 3, ..., n.



    Args:

        n: The length of the list.



    Returns:

        A list of integers from 1 to n.

    """

    return [(i + 1) for i in range(n)]

