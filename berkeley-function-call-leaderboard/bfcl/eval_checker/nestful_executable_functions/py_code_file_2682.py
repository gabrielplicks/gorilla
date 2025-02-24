from typing import List, Optional



def find_first_even_index_with_even_value(A: List[int]) -> Optional[int]:

    """Finds the first index where the current element and the next one are both even.



    Args:

        A: A list of numbers.



    Returns:

        The first index where the current element and the next one are both even, or None if no such index exists.

    """

    for i in range(len(A) - 1):

        if A[i] % 2 == 0 and A[i + 1] % 2 == 0:

            return i

    return None

