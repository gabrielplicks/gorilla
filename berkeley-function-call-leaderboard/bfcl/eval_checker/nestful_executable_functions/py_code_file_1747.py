import re

import typing



def extract_first_number(my_string: str) -> typing.Optional[float]:

    """Extracts the first number from a string, including numbers with decimal points, and converts it to a float data type.

    The function handles both cases when the number is enclosed in parentheses and when the number is not enclosed in parentheses.

    If the number is not present in the string, the function returns None.



    Args:

        my_string: The input string.



    Returns:

        The first number found in the string as a float, or None if no number is found.

    """

    my_string = my_string.replace('(', '').replace(')', '')

    number_pattern = r'\d+\.?\d*'

    number = re.search(number_pattern, my_string)



    if number:

        return float(number.group())

    else:

        return None

