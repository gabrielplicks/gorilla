from typing import AnyStr



def count_character_occurrences(string: AnyStr, character: str) -> int:

    """

    Counts the occurrences of a character in a string using recursion.

    If the character is not found in the string, returns 0.



    Args:

        string: The input string.

        character: The character to count occurrences of.

    """

    if len(string) == 0:

        return 0

    if string[0] == character:

        return 1 + count_character_occurrences(string[1:], character)

    else:

        return count_character_occurrences(string[1:], character)

