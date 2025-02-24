from typing import Union



def round_to_specific_precision(number: float, precision: int) -> str:

    """Rounds a floating-point number to a specific precision.



    Args:

        number: The floating-point number to round.

        precision: The desired precision.

    """

    return str(round(number, precision))

