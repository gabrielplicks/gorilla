from typing import Union



def pad_binary_string(n: Union[int, None], min_length: int = 8) -> str:

    """Converts a given integer to a binary string and pads it with zeroes on the left to ensure a minimum length.



    Args:

        n: The integer to be converted to a binary string.

        min_length: The minimum length of the binary string. Default is 8.



    Returns:

        The padded binary string. If an error occurs, an empty string is returned.

    """

    try:

        binary_string = bin(n).strip("0b")

        padding_length = min_length - len(binary_string)

        if padding_length > 0:

            binary_string = "0" * padding_length + binary_string

        return binary_string

    except TypeError:

        print("TypeError: 'n' must be an integer")

        return ""

