import re

import typing



def parse_numbers(string: str) -> typing.List[float]:

    """Parses a string and returns a list of numeric values.



    Args:

        string: The input string to parse.



    Returns:

        A list of numeric values.

    """

    numbers = re.findall(r'(?<!\d)-?\d+\.?\d*(?:[eE][+-]?\d+)?', string)

    return [float(n) for n in numbers]

