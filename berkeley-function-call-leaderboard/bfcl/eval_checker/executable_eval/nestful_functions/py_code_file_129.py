import re

import typing



def split_string_by_substrings(s: str, delimiters: typing.List[str]) -> typing.List[str]:

    """Splits a string into a list of substrings based on a given set of delimiters.



    Args:

        s: The string to be split.

        delimiters: A list of delimiters used to split the string.

    """

    pattern = "|".join(map(re.escape, delimiters))

    return re.split(pattern, s)

