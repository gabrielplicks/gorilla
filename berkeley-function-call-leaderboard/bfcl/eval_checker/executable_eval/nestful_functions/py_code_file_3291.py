from typing import Union



def int_of_byte(byte: Union[bytes, bytearray]) -> int:

    """Converts a single byte to an integer.

    Args:

        byte: The byte to convert.

    Returns:

        The integer representation of the byte.

    """

    return int.from_bytes(byte, byteorder='big')



def int_from_bytes(bytes: Union[bytes, bytearray], byteorder: str = 'big') -> int:

    """Converts a byte array to an integer.

    Args:

        bytes: The byte array to convert.

        byteorder: The byte order to use.

    Returns:

        The integer representation of the byte array.

    """

    return int.from_bytes(bytes, byteorder)

