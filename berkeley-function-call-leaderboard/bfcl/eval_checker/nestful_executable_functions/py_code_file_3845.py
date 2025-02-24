from typing import List



def get_segment_length(byte_array: List[int], index: int) -> int:

    """Returns the segment length in bytes using bitwise operations.

    Args:

        byte_array: The byte array.

        index: The segment index in the array.

    """

    if index < 0 or index >= len(byte_array):

        return 0

    return byte_array[index] & 0b00001111

