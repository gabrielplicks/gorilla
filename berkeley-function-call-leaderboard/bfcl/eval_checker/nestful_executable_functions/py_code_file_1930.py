import re

import typing



def find_matching_substrings(pattern: str, string: str) -> typing.List[str]:

    """Finds all matching substrings in a larger string using a regex pattern.



    Args:

        pattern: The regex pattern to search for.

        string: The string to search in.



    Returns:

        A list of matching substrings that match the regex pattern.

    """

    regex = re.compile(pattern)

    return regex.findall(string)

