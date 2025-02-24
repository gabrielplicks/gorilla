import re

from typing import Union



def is_matching(string: str, pattern: str) -> bool:

    """Checks whether a given string matches a given wildcard pattern.



    The wildcard pattern can contain the wildcards '?' and '*' (no other characters).

    The '?' wildcard matches any single character, while the '*' wildcard matches

    zero or more characters. The function returns True if the string matches the

    pattern, False otherwise.



    Args:

        string: The string to be matched.

        pattern: The wildcard pattern.

    """

    pattern = pattern.replace('*', '.*').replace('?', '.')

    return bool(re.match(pattern, string))

