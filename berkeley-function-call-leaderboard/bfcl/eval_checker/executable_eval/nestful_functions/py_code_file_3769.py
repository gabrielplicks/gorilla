from typing import List



def extract_first_4_bits(byte_array: bytearray, endian: str = 'big') -> int:

    """Extracts the value encoded in the first 4 bits of a bytearray.

    Args:

        byte_array: The bytearray to extract the bits from.

        endian: The byte ordering. Must be 'big' or 'little'.

    """

    if endian == 'big':

        relevant_bytes = byte_array

    elif endian == 'little':

        relevant_bytes = byte_array[::-1]

    else:

        raise ValueError("Invalid endian: must be 'big' or 'little'")

    relevant_bits = relevant_bytes[0] >> 4

    mask = 15  # 0b1111

    value = relevant_bits & mask

    return value

