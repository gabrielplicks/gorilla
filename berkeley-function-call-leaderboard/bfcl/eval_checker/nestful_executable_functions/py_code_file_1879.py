from typing import Union



def are_all_bits_set(num: Union[int, float]) -> bool:

    """Determines if all the bits in a 32-bit integer are set to 1.



    Args:

        num: The 32-bit integer to check.



    Returns:

        True if all the bits are set to 1, False otherwise.

    """

    mask = 0xFFFFFFFF  # All bits set to 1

    return num & mask == mask

