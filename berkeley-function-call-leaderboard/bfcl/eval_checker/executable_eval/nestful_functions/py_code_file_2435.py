import struct



def int_to_4_byte_array(num: int) -> bytes:

    """Converts an integer to a 4-byte array of bytes, where the most significant byte of the integer is stored in the first byte of the array.



    Args:

        num: The integer to be converted.



    Returns:

        A 4-byte array of bytes representing the integer.

    """

    return struct.pack("!L", num)

