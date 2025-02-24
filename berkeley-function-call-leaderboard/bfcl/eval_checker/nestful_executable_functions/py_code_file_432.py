import re

from typing import Union



def build_regex_pattern(s: str) -> str:

    """Builds a regular expression pattern that matches the pattern in `s` as closely as possible.

    Args:

        s: The input string.

    Returns:

        A regular expression pattern.

    """

    pattern: Union[str, List[str]] = []

    for c in s:

        if c in '<>':

            pattern.append(re.escape(c))

        else:

            pattern.append(c)

    return r"\b" + "".join(pattern) + r"\b"

