from typing import List, Union



def separate_with_hyphens(name: str) -> str:

    """Produces a new string where each letter in the original string `name` is separated by a hyphen (`"-"`).



    Args:

        name: A string containing only letters.



    Returns:

        A new string where each letter in the original string `name` is separated by a hyphen (`"-"`).

    """

    result = ""

    for i, c in enumerate(name):

        result += c

        if i < len(name) - 1:

            result += "-"

    return result

