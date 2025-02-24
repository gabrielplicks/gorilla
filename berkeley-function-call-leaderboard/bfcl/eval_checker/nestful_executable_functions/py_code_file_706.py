from typing import List



def combine_with_separator(strings: List[str], separator: str) -> str:

    """Combines the strings in a list with a separator character.

    Args:

        strings: The list of strings to combine.

        separator: The separator character to use between strings.

    """

    result = ''

    for i, string in enumerate(strings):

        result += string

        if i < len(strings) - 1:

            result += separator

    return result

