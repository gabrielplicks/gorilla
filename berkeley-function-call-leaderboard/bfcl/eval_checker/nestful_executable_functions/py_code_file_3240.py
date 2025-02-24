from typing import Union



def product_sign(a: Union[int, float], b: Union[int, float]) -> int:

    """Calculates the sign of the product of two numbers.



    Args:

        a: The first number.

        b: The second number.



    Returns:

        The sign of the product of the two numbers.

    """

    if a == 0 or b == 0:

        return 0

    elif a > 0 and b > 0:

        return 1

    elif a < 0 and b < 0:

        return 1

    else:

        return -1

