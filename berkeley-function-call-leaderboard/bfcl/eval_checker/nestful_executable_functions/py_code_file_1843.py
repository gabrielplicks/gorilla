from typing import List



def split_str_custom(string: str, separator: str) -> List[str]:

    """Splits a string based on a separator character and returns a list of strings.

    Args:

        string: The input string to be split.

        separator: The separator character.

    """

    result = []

    prev_idx = 0



    for i, char in enumerate(string):

        if char == separator:

            result.append(string[prev_idx:i])

            prev_idx = i + 1



    result.append(string[prev_idx:])

    return result

