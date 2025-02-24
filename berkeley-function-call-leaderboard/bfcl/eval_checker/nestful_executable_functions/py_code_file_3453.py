from typing import Union



def is_text_string(byte_string: Union[bytes, bytearray]) -> bool:

    """Determines if a byte string represents a text string.



    Args:

        byte_string: The input byte string to check.



    Returns:

        A boolean value indicating if the input is a text string or not.

    """

    try:

        byte_string.decode("utf-8")

        return True

    except UnicodeDecodeError:

        return False

