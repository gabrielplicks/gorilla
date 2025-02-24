from typing import List



def split_to_char(str: str, char: str) -> List[str]:

    """Splits a string at a certain character.

    Args:

        str: The input string.

        char: The character to split the string at.

    """

    if char in str:

        return str.split(char)

    else:

        return []

