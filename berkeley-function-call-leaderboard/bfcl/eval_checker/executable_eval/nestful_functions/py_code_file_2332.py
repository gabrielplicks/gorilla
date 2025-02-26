import struct



def integer_to_bytes(integer: int, byteorder: str = "big") -> bytes:

    """

    Converts an integer to its byte representation.

    Args:

        integer: The integer to convert.

        byteorder: The byte order of the resulting byte array, either big-endian (default) or little-endian.

    """

    assert isinstance(integer, int) and 0 <= integer <= 255, "Integer must be between 0 and 255"



    return struct.pack(f"!B" if byteorder == "big" else f"!B", integer)

