import re

from typing import Tuple



def replicate_chars(s: str) -> str:

    """

    Replicates characters in a string based on their numeric value.

    Args:

        s: The input string.

    Returns:

        A string with characters replicated as many times as the numeric value of the character.

    """

    return ''.join(map(lambda x: x[0] * int(x[1]), re.findall(r'(.)(\d*)', s)))

