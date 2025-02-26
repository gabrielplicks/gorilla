import re

from typing import Union



def check_valid_characters(input_str: str) -> Union[bool, None]:

    """

    Checks if the input string contains only valid characters.

    Args:

        input_str: The input string to check.

    Returns:

        True if the string contains valid characters, None otherwise.

    Raises:

        Exception: If the string contains invalid characters.

    """

    if re.match(r'^\w+$', input_str):

        return True

    else:

        raise Exception('Invalid characters found')

