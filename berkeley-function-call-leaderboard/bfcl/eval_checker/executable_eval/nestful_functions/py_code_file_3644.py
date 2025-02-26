import re

from typing import List



def parse_compound_string(s: str) -> List[str]:

    """Parses a compound string into a list of tokens.

    The compound string contains alphanumeric text separated by dots.

    Each token is an alphanumeric string with at most one dot.

    The function returns a list of tokens in the order they appear.

    Args:

        s: The compound string to be parsed.

    """

    return [token.strip() for token in re.split(r'\.+', s) if token.strip()]

