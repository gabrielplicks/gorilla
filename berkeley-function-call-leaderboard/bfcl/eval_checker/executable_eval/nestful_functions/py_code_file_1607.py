import struct



def convert_time_to_uint32(time_in_seconds: int) -> int:

    """Converts a time value in seconds to a big-endian 4-byte unsigned integer representation.



    Args:

        time_in_seconds: The time value in seconds as an integer.



    Returns:

        The big-endian 4-byte unsigned integer representation of the time value.

    """

    return struct.unpack('>I', struct.pack('<I', time_in_seconds))[0]

