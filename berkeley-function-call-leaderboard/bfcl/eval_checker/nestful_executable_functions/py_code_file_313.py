from typing import Union



def round_num(num: Union[float, int], decimal_places: int) -> str:

    """Rounds a numeric value and returns a string representation.

    Args:

        num: The numeric value to be rounded.

        decimal_places: The number of decimal places to round to.

    """

    rounded_value = round(num, decimal_places)

    return f'{rounded_value:.{decimal_places}f}'

