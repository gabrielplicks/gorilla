from typing import Union



def encode_32bit_int(value: Union[int, float]) -> bytearray:

    """Encodes a 32-bit unsigned integer value into a bytearray of 5 bytes.

    The first byte has the first 0-7 bits of the value, the second byte has the next 8-15 bits of the value,

    and so on. The last byte is the final 8 bits of the value.

    Args:

        value: The 32-bit unsigned integer value to be encoded.

    """

    byte_1 = value & 0xFF

    byte_2 = (value >> 8) & 0xFF

    byte_3 = (value >> 16) & 0xFF

    byte_4 = (value >> 24) & 0xFF

    byte_5 = (value >> 32) & 0xFF

    return bytearray([byte_1, byte_2, byte_3, byte_4, byte_5])

