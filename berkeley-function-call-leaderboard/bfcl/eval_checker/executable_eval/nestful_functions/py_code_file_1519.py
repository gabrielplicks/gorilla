import re

import typing



def get_digits_from_string(string: str) -> typing.List[str]:

    """

    Extracts all the digits from a given string and returns them as a list.



    Args:

        string: The input string from which to extract the digits.



    Returns:

        A list of all the digits found in the input string, in the same order as they appear in the string.

    """

    return re.findall(r'\d', string)

