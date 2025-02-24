import re

from typing import List



def filter_hidden_chars(s: str) -> List[str]:

    """

    Filters out hidden characters from a given string.

    Hidden characters are denoted by a leading tilde (`~`) and followed by 1 or more digits.

    For example, `~1` represents a hidden character, while `~123` represents three hidden characters.



    Args:

        s: The input string.



    Returns:

        A list of characters from the input string that are not hidden characters.

    """

    s = re.sub(r'~\d+', '', s)

    filtered_chars = []

    for c in s:

        filtered_chars.append(c)



    return filtered_chars

