import binascii



def string_to_hex_byte_array(string: str) -> bytes:

    """Converts a string to a hexadecimal byte array.

    Args:

        string: The input string to be converted.

    """

    byte_array = string.encode()

    hex_byte_array = binascii.hexlify(byte_array)

    return hex_byte_array



def hex_byte_array_to_string(hex_byte_array: bytes) -> str:

    """Converts a hexadecimal byte array back to a string.

    Args:

        hex_byte_array: The input hexadecimal byte array to be converted.

    """

    byte_array = binascii.unhexlify(hex_byte_array)

    string = byte_array.decode()

    return string

