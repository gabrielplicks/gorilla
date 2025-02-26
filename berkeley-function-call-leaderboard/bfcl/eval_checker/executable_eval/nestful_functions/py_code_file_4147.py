from typing import AnyStr



def get_vowels_count(str: AnyStr) -> int:

    """Calculates the number of vowels in a given string.



    Args:

        str: The input string.



    Returns:

        The number of vowels in the string.

    """

    vowels = 'aeiou'

    count = 0



    for char in str:

        if char in vowels:

            count += 1



    return count

