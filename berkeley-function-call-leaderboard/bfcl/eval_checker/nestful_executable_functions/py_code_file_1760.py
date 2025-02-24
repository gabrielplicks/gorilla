from typing import Union



def mod_range(x: Union[int, float], m: Union[int, float]) -> Union[int, float]:

    """Calculates the value of `x` within the range `[0, m)`.



    Args:

        x: The number to be calculated.

        m: The modulus.

    """

    if x >= 0:

        return x % m

    else:

        remainder = -x % m

        if remainder == 0:

            return 0

        else:

            return m - remainder

