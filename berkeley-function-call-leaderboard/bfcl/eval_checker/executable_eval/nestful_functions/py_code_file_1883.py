from typing import Union



def has_same_sign(x: Union[int, float], y: Union[int, float]) -> bool:

    """

    Returns True if `x` and `y` have the same sign (including zero), and False otherwise.

    Args:

        x: The first number.

        y: The second number.

    """

    if x == 0 or y == 0:

        return True

    elif x < 0 and y < 0:

        return True

    elif x > 0 and y > 0:

        return True

    else:

        return False

