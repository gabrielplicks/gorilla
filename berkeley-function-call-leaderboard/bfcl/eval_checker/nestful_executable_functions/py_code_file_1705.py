from typing import Union



def check_for_binary_power(n: Union[int, float]) -> bool:

    """Checks if an integer is a power of 2.



    Args:

        n: The integer to check.



    Returns:

        True if the integer is a power of 2, otherwise False.

    """

    return n > 0 and (n & (n - 1)) == 0

