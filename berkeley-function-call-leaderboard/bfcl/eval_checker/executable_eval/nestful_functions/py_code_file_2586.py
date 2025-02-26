from typing import List



def bytes_to_binary(byte_list: List[int]) -> str:

    """Converts a list of bytes into a string of bits.



    Each byte is converted into a string of 8 bits and concatenated together.



    Args:

        byte_list: A list of bytes to be converted into a string of bits.



    Returns:

        A string of bits representing the input byte list.

    """

    result = ""

    for byte in byte_list:

        result += format(byte, '08b')

    return result

