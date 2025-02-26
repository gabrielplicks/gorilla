from typing import List



def size_of_X(A: List[bool], B: List[bool]) -> int:

    """Calculates the size of X in bits.

    Args:

        A: A boolean array.

        B: A boolean array of the same size as A.

    """

    X = False

    for i in range(len(A)):

        X = X or (A[i] and B[i])

    return 1  # Size of X in bits

