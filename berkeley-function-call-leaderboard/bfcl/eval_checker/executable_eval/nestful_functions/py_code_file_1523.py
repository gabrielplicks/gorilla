import binascii



def hex_from_bytes(byte_array: bytes) -> str:

    """Converts a byte array to its hexadecimal representation.



    Args:

        byte_array: The byte array to convert.



    Returns:

        The hexadecimal representation of the byte array as a string.

    """

    return binascii.hexlify(byte_array).decode('utf-8')

