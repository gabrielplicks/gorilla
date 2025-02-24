import itertools

from typing import List



def generate_permutation_list(n: int) -> List[List[int]]:

    """Generates a list of all possible permutations of `n` numbers.



    Args:

        n: The number of elements in the permutations.



    Returns:

        A list of lists, where each list represents a single permutation.

    """

    return [list(permutation) for permutation in itertools.permutations(range(1, n + 1))]

