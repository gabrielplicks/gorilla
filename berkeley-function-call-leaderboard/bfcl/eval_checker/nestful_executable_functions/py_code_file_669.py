import re

import typing



def create_pattern(strings: typing.List[str]) -> str:

    """Creates a regular expression pattern that matches any string that contains all the specified strings.



    Args:

        strings: A list of strings to include in the pattern.

    """

    pattern = '.*' + '.*'.join(re.escape(s) for s in strings) + '.*'

    return pattern

