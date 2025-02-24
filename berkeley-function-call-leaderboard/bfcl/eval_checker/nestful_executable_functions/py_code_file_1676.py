from typing import List



def remove_spaces(string: str) -> str:

    """Removes all spaces from a given string and returns the resulting string.



    Args:

        string: The input string.



    Returns:

        The input string with all spaces removed.

    """

    result: List[str] = []

    for char in string:

        if char != ' ':

            result.append(char)

    return ''.join(result)

