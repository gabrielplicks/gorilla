from typing import Union



def count_ones_bit(num: Union[int, float]) -> int:

    """Calculates the number of 1-bits in the binary representation of a number using bit manipulation and a for loop.



    Args:

        num: The number to count the 1-bits in its binary representation.



    Returns:

        The total count of 1-bits in the binary representation of the number.

    """

    count = 0

    mask = 1

    for _ in range(32):

        if num & mask:

            count += 1

        mask = mask << 1

    return count

