from typing import Union



def check_divisible(n: Union[int, float]) -> bool:

    """Checks if an integer `n` is divisible by `3` or `5`.



    Args:

        n: The integer to check for divisibility.



    Returns:

        True if `n` is divisible by either `3` or `5`, and False otherwise.

    """

    return n % 3 == 0 or n % 5 == 0

