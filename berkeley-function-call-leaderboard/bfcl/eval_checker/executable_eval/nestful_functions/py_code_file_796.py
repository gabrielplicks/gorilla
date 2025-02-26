import re

from typing import List



def decode_from_alphanum(text: str) -> str:

    """Converts a string of alphanumeric characters to a string of digits.



    Args:

        text: A string of alphanumeric characters.



    Returns:

        A string consisting of only digits.

    """

    alphanum_pattern = re.compile(r'\d+')

    digits: List[str] = re.findall(alphanum_pattern, text)

    return ''.join(digits)

