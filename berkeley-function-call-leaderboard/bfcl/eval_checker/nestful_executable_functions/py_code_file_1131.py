import re

import typing



def find_surrounded_substring(string: str, first_char: str, second_char: str) -> typing.Optional[str]:

    """Finds the substring in a given string that is surrounded by a pair of specified characters.



    Args:

        string: The input string.

        first_char: The first character of the pair.

        second_char: The second character of the pair.



    Returns:

        The substring surrounded by the pair of characters, or None if no such substring is found.

    """

    pattern = f'(?<={first_char})[^{first_char}{second_char}]*(?={second_char})'

    match = re.search(pattern, string)

    if match:

        return match.group()

    else:

        return None

