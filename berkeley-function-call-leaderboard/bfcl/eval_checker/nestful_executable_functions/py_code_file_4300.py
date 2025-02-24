def hexadecimal_to_bytes(hex_string: str) -> bytes:

    """

    Converts a hexadecimal string to a byte string.

    Args:

        hex_string: The hexadecimal string to convert.

    """

    if len(hex_string) % 2 != 0:

        raise ValueError("Invalid hexadecimal string: length is not even")

    for char in hex_string:

        if char not in '0123456789abcdefABCDEF':

            raise ValueError(f"Invalid hexadecimal string: invalid character '{char}'")

    byte_string = b''

    for i in range(0, len(hex_string), 2):

        byte_int = int(hex_string[i:i+2], 16)

        byte_string += bytes([byte_int])



    return byte_string

