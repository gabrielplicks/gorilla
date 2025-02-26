import re

from typing import Union



def base5_string(n: Union[int, str]) -> Union[str, str]:

    """

    Converts an integer to a string representing the integer in base 5, with the digits in the string arranged from most significant to least significant.

    The digits in the base 5 string should be integers between 0 and 4.

    If the input is not an integer, return "Invalid input".



    Args:

        n: The integer to be converted to base 5.



    Returns:

        A string representing the integer in base 5.

    """

    if not isinstance(n, int) or not re.match(r'^[1-9]\d*$', str(n)):

        return "Invalid input"

    digits = []

    while n:

        n, remainder = divmod(n, 5)

        digits.append(str(remainder))

    return ''.join(reversed(digits))

