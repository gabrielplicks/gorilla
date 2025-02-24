import re

import typing



def extract_last_match(string: str, pattern: str, delimiter: typing.Optional[str] = None) -> typing.Union[str, typing.List[str]]:

    """Extracts the last match of a given pattern in a string.



    If the delimiter is not specified, the last match of the pattern should be returned.

    If the delimiter is specified, the entire string, including the last match, should be returned.



    Args:

        string: The input string.

        pattern: The pattern to search for.

        delimiter: The delimiter to separate the end of the string from the match.

    """

    if delimiter is None:

        matches = re.findall(pattern, string)

        return matches[-1]

    else:

        substrings = string.split(delimiter)

        last_substring = substrings[-1]

        matches = re.findall(pattern, last_substring)

        return matches[-1]

