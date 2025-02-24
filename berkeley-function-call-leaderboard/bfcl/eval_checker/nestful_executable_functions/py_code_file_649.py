import struct



def encode_as_little_endian(n: int) -> bytes:

    """Converts an integer to its little-endian encoding, i.e., the least significant byte (LSB) is first.



    Args:

        n: The integer to be converted.



    Returns:

        The little-endian encoding of the integer.

    """

    return struct.pack("<I", n)

