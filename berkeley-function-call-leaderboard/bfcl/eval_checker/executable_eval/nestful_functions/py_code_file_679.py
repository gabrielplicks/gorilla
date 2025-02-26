from typing import Set



def number_of_unique_characters(string: str) -> int:

    """Returns the number of unique characters in a string, where each character is represented as a byte.

    Args:

        string: The input string.

    """

    unique_characters: Set[bytes] = set()

    for char in string:

        unique_characters.add(char.encode())

    return len(unique_characters)

