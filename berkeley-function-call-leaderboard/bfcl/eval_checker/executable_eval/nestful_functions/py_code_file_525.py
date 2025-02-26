from typing import Union



def signed_int_to_big_endian(num: Union[int, float]) -> str:

    """Converts a signed integer to its big endian binary representation.



    Args:

        num: The signed integer to convert.



    Returns:

        The big endian binary representation of the signed integer as a string.

    """

    unsigned_num = num & 0xffffffff  # Convert to unsigned int

    return format(unsigned_num, '032b')  # Pad with zeros to 32 bits

