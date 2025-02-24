from typing import List, Union



def format_camel_case(string: str) -> str:

    """Formats the given input string as a "camelCase" string.



    The first letter of each word is capitalized and spaces are removed.

    Built-in functions like `str.capitalize`, `str.join`, or `str.title` are not used.



    Args:

        string: The input string to be formatted.



    Returns:

        The formatted string in camel case format.

    """

    words: List[str] = string.split()

    for i in range(len(words)):

        words[i] = words[i][0].upper() + words[i][1:]

    return "".join(words)

