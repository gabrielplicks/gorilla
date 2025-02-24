from typing import AnyStr



def count_consecutive_pairs(s: AnyStr) -> int:

    """Calculates the number of consecutive pairs of letters in a string.



    Args:

        s: The input string.



    Returns:

        The number of consecutive pairs of letters in the string.

    """

    if not s:

        return 0



    consecutive_pairs = 0



    for i in range(len(s) - 1):

        if s[i] == s[i + 1]:

            consecutive_pairs += 1



    return consecutive_pairs

