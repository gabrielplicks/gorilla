import re

import typing



def replace_program(text: str) -> str:

    """Replaces all occurrences of "program" with "programmer" in the input string using regular expressions.



    Args:

        text: The input string.



    Returns:

        The modified string.

    """

    return re.sub(r"program\s*er?", "programmer", text)

