import re

import typing



def count_matches_regex(text: str, pattern: str) -> int:

    """Counts the number of matches found in the string using the given regular expression pattern.

    Args:

        text: The input string to search for matches.

        pattern: The regular expression pattern to search for in the string.

    """

    return len(re.findall(pattern, text))

