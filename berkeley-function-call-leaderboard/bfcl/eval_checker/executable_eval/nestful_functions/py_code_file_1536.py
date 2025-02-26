from typing import List



def invert_permutation(permutation: List[int]) -> List[int]:

    """Inverts a permutation of integers.



    Args:

        permutation: A permutation of integers from 0 to the length of the array minus 1.



    Returns:

        A new array which inverts the permutation.

    """

    inverted_permutation = [0] * len(permutation)



    for index, element in enumerate(permutation):

        inverted_permutation[element] = index



    return inverted_permutation

