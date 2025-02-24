from typing import Union



def create_binary_mask(position: Union[int, float]) -> int:

    """Creates a binary mask of width 16 with a set bit at a specified position.



    Args:

        position: The position of the set bit.

    """

    bit_index = 2 ** position

    binary_mask = bit_index



    return binary_mask

