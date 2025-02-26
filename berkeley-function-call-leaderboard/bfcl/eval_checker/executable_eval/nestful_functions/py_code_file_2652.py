from typing import Tuple



def get_div_mod(a: int, b: int) -> Tuple[int, int]:

    """Calculates the quotient and remainder of the integer division of a and b.



    Args:

        a: The dividend.

        b: The divisor.



    Raises:

        ZeroDivisionError: If b is zero.

    """

    if b == 0:

        raise ZeroDivisionError("Cannot divide by zero")



    return a // b, a % b

