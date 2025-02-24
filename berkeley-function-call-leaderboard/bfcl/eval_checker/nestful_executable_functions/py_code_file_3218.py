from typing import Union



def round_to_nearest_base(num: int, base: Union[int, None] = 10) -> int:

    """Rounds a number to the nearest multiple of a given base.



    Args:

        num: The number to round.

        base: The base to round to. Default is 10.

    """

    remainder = num % base

    if remainder < base / 2:

        return num - remainder

    else:

        return num + base - remainder

