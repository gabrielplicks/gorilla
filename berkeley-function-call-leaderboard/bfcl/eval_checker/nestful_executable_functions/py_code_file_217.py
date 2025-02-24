from typing import Union



def count_consecutive_zeros(n: Union[int, float]) -> int:

    """Counts the number of consecutive 0s in the binary representation of an integer n.



    Args:

        n: The integer to count the consecutive 0s in its binary representation.



    Returns:

        The number of consecutive 0s in the binary representation of n.

    """

    binary_representation = bin(n)[2:]

    consecutive_zeros = 0

    for char in binary_representation:

        if ord(char) != 48:

            consecutive_zeros = 0

        else:

            consecutive_zeros += 1

    return consecutive_zeros

