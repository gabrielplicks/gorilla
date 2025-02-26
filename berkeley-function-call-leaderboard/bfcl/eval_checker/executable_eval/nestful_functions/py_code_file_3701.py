import re

import typing



def parse_string_to_floats(string: str) -> typing.List[float]:

    """Parses a string that may contain integer or float values to a list of float values.

    Args:

        string: The input string to parse.

    Returns:

        A list of float values, with invalid values ignored.

    """

    pattern = re.compile(r'\d+\.?\d*')

    values = pattern.findall(string)

    return [float(value) for value in values if value.replace('.', '').isdigit()]

