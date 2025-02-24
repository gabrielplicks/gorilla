from typing import Union



def byte_string_to_bit_string(byte_string: Union[bytes, bytearray]) -> str:

    """Converts a byte-string to a bit-string using bitwise operations.



    Args:

        byte_string: The byte-string to convert.



    Returns:

        The bit-string representation of the byte-string.

    """

    integer = int.from_bytes(byte_string, 'little')

    bit_string = ''

    for i in range(8):

        bit_string += str((integer >> i) & 1)



    return bit_string[::-1]

