from typing import Tuple



def is_lexicographically_smaller(s1: str, s2: str) -> bool:

    """Determines whether s1 is lexicographically smaller than s2.



    Args:

        s1: The first string to compare.

        s2: The second string to compare.



    Returns:

        True if s1 is lexicographically smaller than s2, False otherwise.

    """

    for i in range(min(len(s1), len(s2))):

        if s1[i] < s2[i]:

            return True

        elif s2[i] < s1[i]:

            return False



    if len(s1) < len(s2):

        return True

    else:

        return False

