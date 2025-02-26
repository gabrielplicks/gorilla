import re

import typing



def match_all_tags(string: str) -> typing.List[str]:

    """

    Returns a list of all HTML tags found in the input string.

    Args:

        string: The input string to search for HTML tags.

    """

    pattern = r"<([a-zA-Z0-9]+)>"

    matches = re.findall(pattern, string)

    return matches

