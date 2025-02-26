import zlib



def hash_crc32(s: str) -> int:

    """Calculates the CRC-32 hash of a string `s`.



    Args:

        s: The input string.



    Returns:

        The CRC-32 hash of the input string as an integer.

    """

    return zlib.crc32(s.encode())

