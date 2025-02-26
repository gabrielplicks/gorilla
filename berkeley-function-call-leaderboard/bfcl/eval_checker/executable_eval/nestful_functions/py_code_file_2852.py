import re

from typing import List



def split_by_separators(s: str, separators: List[str]) -> List[str]:

    """

    Splits a string by one or more separators and returns a list of substrings.

    Empty substrings are ignored.



    Args:

        s: The input string.

        separators: A list of separators.



    Returns:

        A list of substrings.

    """

    regex_pattern = "|".join(re.escape(sep) for sep in separators)

    return [substr for substr in re.split(regex_pattern, s) if substr != ""]

