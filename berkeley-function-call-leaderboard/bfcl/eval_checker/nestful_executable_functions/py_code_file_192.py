from typing import Tuple



def count_differences(str1: str, str2: str) -> int:

    """Calculates the number of characters that are different between two strings.

    Args:

        str1: The first string.

        str2: The second string.

    Returns:

        The number of different characters between the two strings, or `-1` if the strings are not of equal length.

    """

    if len(str1) != len(str2):

        return -1



    difference_count = 0

    for i in range(len(str1)):

        if str1[i] != str2[i]:

            difference_count += 1



    return difference_count

