from typing import Set



def replace_first_repeated_character(string: str) -> str:

    """Replaces the first occurrence of a repeating character in a string with a dash ('-').

    If no repeating characters are found, returns the original string.

    Args:

        string: The input string.

    """

    unique_characters: Set[str] = set()

    for i, char in enumerate(string):

        if char in unique_characters:

            string = string[:i] + '-' + string[i+1:]

            break

        else:

            unique_characters.add(char)

    return string

