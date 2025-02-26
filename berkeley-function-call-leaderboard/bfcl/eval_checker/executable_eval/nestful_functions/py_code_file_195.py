from typing import List



def int_list_to_hex(int_list: List[int]) -> str:

    """Converts a list of integers into a hex string representation.

    Each integer is an ASCII character code, and the function returns a string that represents the ASCII values as a hexadecimal.

    Args:

        int_list: A list of integers.

    """

    hex_str = ""

    for i in int_list:

        hex_str += hex(i)

    return hex_str

