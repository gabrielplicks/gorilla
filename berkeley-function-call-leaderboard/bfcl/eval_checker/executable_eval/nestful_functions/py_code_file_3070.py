from typing import List



def is_rotation_array(A: List[int], B: List[int]) -> bool:

    """Checks if B is a rotation of A.



    Args:

        A: The first list.

        B: The second list.



    Returns:

        True if B is a rotation of A, False otherwise.

    """

    if len(A) != len(B):

        return False



    n = len(A)

    for i in range(n):

        if A[i % n] == B[0]:

            j = 0

            while j < n and A[(i + j) % n] == B[j]:

                j += 1

            if j == n:

                return True

    return False

