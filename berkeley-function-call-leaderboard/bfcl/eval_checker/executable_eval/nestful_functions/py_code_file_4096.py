from typing import Union



def decode_utf8_string(byte_string: Union[bytes, str]) -> Union[str, None]:

    """Decodes a byte string encoded in UTF-8 to a Python string.

    Handles different Python versions (2.7 and 3.x) and possible exceptions related to encoding errors.

    Args:

        byte_string: The byte string to be decoded.

    Returns:

        The decoded string, or None if an error occurred.

    """

    try:

        return byte_string.decode('utf-8', errors='ignore')

    except UnicodeDecodeError as e:

        print(f"Error: {e}")

        return None

