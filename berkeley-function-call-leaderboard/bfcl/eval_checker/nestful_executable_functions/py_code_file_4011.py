from typing import Union



def byte_to_hex(bytes_str: Union[bytes, bytearray]) -> str:

    """Converts a byte string to a colon-delimited hex string representation.

    Args:

        bytes_str: The byte string to convert.

    """

    return ":".join(f"{b:x}" for b in bytes_str)

