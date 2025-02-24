from itertools import permutations

from typing import List



def check_abc(s: str) -> bool:

    """Checks if a string contains the substring 'abc' in any order.



    Args:

        s: The input string.



    Returns:

        True if the string contains 'abc' in any order, False otherwise.

    """

    for permutation in permutations('abc'):

        if ''.join(permutation) in s:

            return True

    return False

