from typing import Union



def divide_if_divisible(a: int, b: int) -> Union[float, int]:

    """Divides `a` by `b` if `a` is divisible by `b`, otherwise returns `0`.



    Args:

        a: The numerator.

        b: The denominator.

    """

    if a % b == 0:

        return a / b

    else:

        return 0

