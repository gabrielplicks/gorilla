from typing import Optional



def first_non_repeated_character(string: str) -> Optional[str]:

    """

    Returns the first non-repeated character in a string.



    Args:

        string: The input string.



    Returns:

        The first non-repeated character in the string, or None if there is no non-repeated character.

    """

    for i in range(len(string)):

        if string.count(string[i]) == 1:

            return string[i]

    return None

