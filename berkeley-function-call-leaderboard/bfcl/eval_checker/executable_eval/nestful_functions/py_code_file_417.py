import re

from typing import List



def split_string_regex(s: str) -> List[str]:

    """Splits an input string `s` into a list of strings using a regex.

    Args:

        s: The input string to split.

    Returns:

        A list of strings, excluding any empty strings.

    """

    return [x for x in re.split(r"\s+", s) if x]

