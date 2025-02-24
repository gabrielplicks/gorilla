from typing import Tuple



def remove_subsequence(string: str, start_end: Tuple[str, str]) -> str:

    """Removes all subsequences between two specific characters in a string.



    Args:

        string: The input string.

        start_end: A tuple of two characters representing the start and end of the subsequence to remove.



    Returns:

        The modified string, excluding the subsequence between the characters.

    """

    start, end = start_end

    output = ""

    inside_subsequence = False



    for char in string:

        if char == start:

            inside_subsequence = True

        elif char == end:

            inside_subsequence = False

        elif not inside_subsequence:

            output += char



    return output

