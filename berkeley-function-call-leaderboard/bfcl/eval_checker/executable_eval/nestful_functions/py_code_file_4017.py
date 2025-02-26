import re

import typing



def extract_numeric_values(input_string: str) -> typing.List[typing.Union[int, float]]:

    """Extracts numeric values from a given input string.



    Args:

        input_string: The input string to extract numeric values from.



    Returns:

        A list of integers and floats representing the numeric values found in the input string.

    """

    numeric_pattern = r'(\d+\.?\d*)'



    numeric_values = []

    for match in re.finditer(numeric_pattern, input_string):

        value = match.group(1)

        if value.isdigit():

            value = int(value)

        else:

            value = float(value)

        numeric_values.append(value)



    return numeric_values

