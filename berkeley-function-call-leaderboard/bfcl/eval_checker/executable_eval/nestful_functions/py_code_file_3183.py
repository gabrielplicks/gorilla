from typing import Union



def is_negative(n: int) -> bool:

    """Checks if the integer is negative.

    Args:

        n: The integer to check.

    """

    return n < 0



def is_zero(n: int) -> bool:

    """Checks if the integer is zero.

    Args:

        n: The integer to check.

    """

    return n == 0



def is_positive(n: int) -> bool:

    """Checks if the integer is positive.

    Args:

        n: The integer to check.

    """

    return n > 0



def return_sign(n: int) -> Union[int, float]:

    """Returns the sign of the integer.

    Args:

        n: The integer to check.

    """

    if is_negative(n):

        return -1

    elif is_zero(n):

        return 0

    else:

        return 1

