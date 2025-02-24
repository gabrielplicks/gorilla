from typing import Union



def norm_sign(num: Union[int, float]) -> int:

    """Calculates the normalized sign of the unit magnitude of a number.

    Args:

        num: The input number.

    """

    if num >= 0:

        return 1

    else:

        return -1

