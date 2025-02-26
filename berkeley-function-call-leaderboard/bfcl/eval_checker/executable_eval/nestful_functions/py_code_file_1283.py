from typing import Optional



def swap_first_last_characters(string: str) -> Optional[str]:

    """Swaps the first and last characters of a string.



    Args:

        string: The input string.



    Returns:

        A string where the first and last characters are swapped, except if the string has less than two characters.

    """

    if len(string) < 2:

        return string

    else:

        return string[-1] + string[1:-1] + string[0]

