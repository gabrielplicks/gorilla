from typing import Union



def sign(x: Union[int, float]) -> int:

    """Returns the sign of a number.

    Args:

        x: The number to find the sign of.

    """

    if x > 0:

        return 1

    elif x < 0:

        return -1

    else:

        return 0

