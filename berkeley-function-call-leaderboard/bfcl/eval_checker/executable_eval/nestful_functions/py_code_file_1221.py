from typing import Tuple



def my_divmod(x: int, y: int) -> Tuple[int, int]:

    """Returns a tuple containing the quotient and the remainder of x divided by y.



    Args:

        x: The dividend.

        y: The divisor.



    Raises:

        TypeError: If either x or y is not an integer.

    """

    if not isinstance(x, int) or not isinstance(y, int):

        raise TypeError("x and y must be integers")

    return divmod(x, y)

