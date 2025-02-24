from typing import Tuple



def div_and_mod(x: int, y: int) -> Tuple[int, int]:

    """

    Returns the quotient and remainder of x divided by y.



    Args:

        x: The dividend.

        y: The divisor.



    Raises:

        TypeError: If either of x or y is not an integer.

    """

    if not isinstance(x, int) or not isinstance(y, int):

        raise TypeError("Both arguments must be integers.")

    q = x // y

    r = x % y

    return q, r

