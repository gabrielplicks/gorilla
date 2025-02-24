from typing import Union



def extract_digits_from_string(input_string: Union[str, bytes]) -> str:

    """Extracts all the digits from a given string.

    Args:

        input_string: The input string to extract digits from.

    Returns:

        The extracted digits as a string.

    """

    return "".join(char for char in input_string if char.isdigit())

