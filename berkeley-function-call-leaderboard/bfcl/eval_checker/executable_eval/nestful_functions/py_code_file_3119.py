from typing import List



def find_intersection(A: List[int], B: List[int]) -> List[int]:

    """Finds the intersection of two lists using dynamic programming to store results and avoid repeated computation.



    Args:

        A: The first list.

        B: The second list.



    Returns:

        The intersection of the two lists as a new list.

    """

    cache = {}



    def intersection_helper(A: List[int], B: List[int]) -> List[int]:

        key = str(A) + str(B)

        if key in cache:

            return cache[key]

        result = []

        for a in A:

            if a in B:

                result.append(a)

        cache[key] = result

        return result

    return intersection_helper(A, B)

