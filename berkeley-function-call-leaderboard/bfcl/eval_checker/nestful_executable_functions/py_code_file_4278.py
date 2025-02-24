import re

import typing



def sum_integer_literals(s: str) -> int:

    """Calculates the sum of all integer literals in the input string `s` in their original numeric form.

    Args:

        s: The input string containing integer literals.

    """

    pattern = r"(\-?\d+)"

    matches = re.findall(pattern, s)

    sum_literals = sum(int(m) for m in matches)

    return sum_literals

