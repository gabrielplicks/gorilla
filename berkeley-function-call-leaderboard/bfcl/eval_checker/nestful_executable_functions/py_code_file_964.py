import re

import typing



def extract_positive_ints(text: str) -> typing.List[int]:

    """Extracts all positive integers from the input string.



    Args:

        text: The input string to parse.



    Returns:

        A list of positive integers found in the input string.

    """

    pattern = re.compile(r'\d+')

    matches = pattern.findall(text)

    return [int(match) for match in matches]

