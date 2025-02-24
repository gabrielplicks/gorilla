import re

from typing import List



def is_strictly_ascending(s: str) -> bool:

    """Checks if the digits in a string are strictly ascending.



    Args:

        s: The input string.



    Returns:

        True if the digits in the string are strictly ascending, False otherwise.

    """

    if len(s) == 0 or len(set(s)) == 1:

        return True

    digits: List[int] = [int(digit) for digit in s if digit.isdigit()]

    for i in range(1, len(digits)):

        if digits[i] <= digits[i - 1]:

            return False



    return True

