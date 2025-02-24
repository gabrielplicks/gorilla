import re

import typing



def find_matches(strings: typing.List[str], pattern: str) -> typing.List[str]:

    """Finds all strings in a given list that match a given regular expression pattern.



    Args:

        strings: A list of strings to search for matches.

        pattern: The regular expression pattern to match.



    Returns:

        A list of matching strings. If no matches are found, an empty list is returned.

    """

    matches = []



    for string in strings:

        if re.search(pattern, string):

            matches.append(string)



    return matches

