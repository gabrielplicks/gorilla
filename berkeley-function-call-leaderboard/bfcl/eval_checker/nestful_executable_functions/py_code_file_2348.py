from typing import List



def first_k(L: List[int], k: int) -> List[int]:

    """Returns the first k elements of a list L if k is less than or equal to the length of L. Otherwise, returns L.

    Args:

        L: A list of integers.

        k: The number of elements to return.

    """

    if k <= len(L):

        return L[:k]

    else:

        return L



def first_k(L: List[int], k: int) -> List[int]:

    """Returns the first k elements of a list L if k is less than or equal to the length of L. Otherwise, returns L.

    Args:

        L: A list of integers.

        k: The number of elements to return.

    """

    return [L[i] for i in range(min(k, len(L)))]

