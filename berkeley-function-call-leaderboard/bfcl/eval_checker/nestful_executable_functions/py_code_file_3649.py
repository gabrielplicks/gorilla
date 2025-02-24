import struct



def pack_int_to_little_endian(num: int) -> str:

    """Packs a 32-bit unsigned integer into a little-endian byte array.



    Args:

        num: The integer to pack. Must be a 32-bit unsigned integer in hexadecimal format with leading zeroes.



    Returns:

        A string of the packed bytes in hexadecimal format.

    """

    assert isinstance(num, int) and 0 <= num <= 0xffffffff, "Invalid number"

    packed_bytes = struct.pack("<L", num)

    return packed_bytes.hex()

