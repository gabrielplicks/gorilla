from typing import Union



def int2bin(num: Union[int, float]) -> str:

    """Prints an integer as a binary string.



    Args:

        num: The integer to be converted to a binary string.

    """

    bin_str = bin(num)[2:]  # Convert to binary string and slice the first two characters

    return '0b{}'.format(bin_str)

