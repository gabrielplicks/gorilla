from typing import List



def int_to_byte_array(int_num: int) -> List[int]:

    """Converts an integer into its big-endian byte representation.



    Args:

        int_num: The integer to convert.



    Returns:

        A byte array representing the integer.

    """

    byte_array = bytearray(8)

    unsigned_int_num = int_num & 0xFFFFFFFFFFFFFFFF

    for i in range(8):

        byte_array[i] = (unsigned_int_num >> (56 - i*8)) & 0xFF

    return byte_array

