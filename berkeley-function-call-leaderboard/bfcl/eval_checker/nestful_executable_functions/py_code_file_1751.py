from typing import Union



def divisor_bits(n: int, divisor: Union[int, float] = 2) -> bool:

    """Checks if a number is divisible by another number.



    Args:

        n: The number to be checked for divisibility.

        divisor: The number by which n is to be divided. Defaults to 2.

    """

    if divisor == 0:

        return False

    if divisor > 0:

        return n % divisor == 0

    return n % abs(divisor) == 0

