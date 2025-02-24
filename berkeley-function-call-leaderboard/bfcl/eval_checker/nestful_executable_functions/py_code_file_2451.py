from typing import Union



def round_to_decimal_places(num: Union[float, int], d: int) -> float:

    """Rounds a floating-point number to a specific number of decimal places.



    Args:

        num: The number to be rounded.

        d: The number of decimal places to round to.

    """

    num_str = str(num)

    decimal_index = num_str.find('.')

    multiplier = 10 ** d

    num_multiplied = num * multiplier

    rounded = round(num_multiplied)

    result = rounded / multiplier



    return result

