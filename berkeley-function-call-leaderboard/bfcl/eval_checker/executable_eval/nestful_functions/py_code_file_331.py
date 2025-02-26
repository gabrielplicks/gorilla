from typing import List



def count_to_n(n: int) -> List[int]:

    """Creates a list of integers from 0 to `n` (inclusive).



    Args:

        n: The upper limit of the list.



    Returns:

        A list of integers from 0 to `n` (inclusive).

    """

    numbers = []

    for i in range(n+1):

        numbers.append(i)

    return numbers

