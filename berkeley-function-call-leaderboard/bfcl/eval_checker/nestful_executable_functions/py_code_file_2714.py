from typing import List



def int_to_binary_list(num: int) -> List[int]:

    """Converts a positive integer to a list of 0's and 1's representing its binary value.

    Args:

        num: The positive integer to be converted.

    """

    binary_str = bin(num)[2:]

    return [int(bit) for bit in binary_str]

