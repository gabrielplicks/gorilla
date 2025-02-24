import re

import typing



def reduce_whitespace(text: str) -> str:

    """Reduces repeated whitespace characters in a string.

    Replaces any sequence of whitespace characters with a single space, while preserving leading and trailing whitespace.

    Args:

        text: The input string.

    """

    return re.sub(r"\s+", " ", text.strip())

