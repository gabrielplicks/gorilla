from typing import Union



def binary_set_bits(num: Union[int, float]) -> int:

    """Calculates the number of set bits in the binary representation of a positive integer.



    Args:

        num: A positive integer or float. If a float is provided, it will be converted to an integer.



    Returns:

        The number of set bits in the binary representation of the input number.

    """

    binary = bin(int(num))

    return binary.count('1')

