from typing import Union



def count_bits(num: Union[int, float]) -> int:

    """Counts the number of 1 bits in the binary representation of a given number.



    Args:

        num: The number to count the bits of.



    Returns:

        The number of 1 bits in the binary representation of the given number.

    """

    return bin(int(num)).count("1")

