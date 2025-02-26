import re

import typing



def consecutive_sequences(input_string: str) -> typing.List[str]:

    """

    Searches for consecutive sequences of digits in a string.

    Args:

        input_string: The input string to search for consecutive sequences of digits.

    Returns:

        A list of strings containing consecutive sequences of digits found in the input string.

    """

    try:

        assert isinstance(input_string, str), "Input must be a string"

        consecutive_digits = re.findall(r"\d{2,}", input_string)

        return consecutive_digits

    except AssertionError:

        return []

