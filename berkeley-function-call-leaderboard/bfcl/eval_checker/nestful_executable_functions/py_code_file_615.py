from typing import List



def hex_to_bytes(hex_str: str) -> bytes:

    """Converts a string of hexadecimal values to the corresponding byte array.



    Args:

        hex_str: A string of hexadecimal values. The input string is guaranteed to be of even length.



    Raises:

        ValueError: If the input string does not have an even length.

    """

    if len(hex_str) % 2 != 0:

        raise ValueError("Invalid input: the input string should have an even length.")



    byte_list: List[int] = []

    for i in range(0, len(hex_str), 2):

        byte_value = int(hex_str[i:i+2], 16)

        byte_list.append(byte_value)



    return bytes(byte_list)

