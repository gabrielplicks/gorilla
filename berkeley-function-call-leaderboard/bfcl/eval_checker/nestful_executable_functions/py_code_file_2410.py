import math



def pad_integer(n: int, d: int) -> int:

    """Pads an integer so that it is a multiple of another integer.



    Args:

        n: The integer to pad.

        d: The divisor.

    """

    if n % d == 0:

        return n

    nearest_multiple = math.ceil(n / d) * d

    return nearest_multiple

