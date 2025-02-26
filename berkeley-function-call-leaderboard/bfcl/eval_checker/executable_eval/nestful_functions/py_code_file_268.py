from typing import Tuple



def max_elements(block_size: int, element_size: int) -> int:

    """Calculates the maximum number of elements that can be stored in a block.



    Args:

        block_size: The size of the block.

        element_size: The size of each element.



    Returns:

        The maximum number of elements that can be stored in the block.

    """

    return block_size // element_size

