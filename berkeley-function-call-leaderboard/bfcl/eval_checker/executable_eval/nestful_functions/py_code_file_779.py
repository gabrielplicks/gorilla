from typing import Union



def convert_binary_to_integer(binary_string: str) -> int:

    """Converts a binary string to an integer.

    Args:

        binary_string: The binary string to convert.

    """

    return int(binary_string, 2)



def parse_binary_string(binary_string: Union[str, bytes]) -> int:

    """Parses a binary string and returns the integer value.

    Args:

        binary_string: The binary string to parse.

    """

    binary_string = binary_string.strip()

    if not binary_string.startswith('0b'):

        binary_string = '0b' + binary_string

    return int(binary_string, 2)

