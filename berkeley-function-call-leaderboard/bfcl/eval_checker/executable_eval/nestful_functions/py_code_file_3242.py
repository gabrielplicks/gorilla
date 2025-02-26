import re

import typing



def extract_consecutive_substrings(s: str) -> typing.List[str]:

    """Extracts all consecutive substrings of the given string that match the regular expression.



    Args:

        s: The input string.



    Returns:

        A list of all consecutive substrings that match the regular expression.

    """

    pattern = r'(?i)[a-z]+[a-z0-9]*[a-z][a-z0-9]+'

    return re.findall(pattern, s)

