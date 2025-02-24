from typing import AnyStr



def get_encoding_count(input_string: AnyStr) -> int:

    """Calculates the number of bytes required to encode the input string using the UTF-8 encoding.



    Args:

        input_string: The input string to be encoded.



    Returns:

        The number of bytes required to encode the input string using the UTF-8 encoding.

    """

    return len(input_string.encode('utf-8'))

