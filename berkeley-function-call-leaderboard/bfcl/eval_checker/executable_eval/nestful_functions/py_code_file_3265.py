from typing import Union



import math



def dB_to_linear(dB: Union[int, float]) -> float:

    """Converts decibel to linear units.

    Args:

        dB: A floating-point number representing a decibel value.

    """

    linear = 10 ** (dB / 20)

    return linear

