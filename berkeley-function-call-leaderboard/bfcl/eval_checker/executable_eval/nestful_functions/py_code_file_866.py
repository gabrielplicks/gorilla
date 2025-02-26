from typing import Tuple



def floor_divide(a: int, b: int) -> int:

    """Calculates the floor of `a` divided by `b`.



    Args:

        a: The dividend.

        b: The divisor.



    Raises:

        ZeroDivisionError: If `b` is `0`.

    """

    if b == 0:

        raise ZeroDivisionError("The divisor cannot be 0.")

    return a // b

