from typing import Union



def decimal_to_percentage(decimal: Union[int, float]) -> int:

    """Converts a decimal number between 0 and 1 to a percentage (rounded to an integer).



    Args:

        decimal: The decimal number to convert.



    Returns:

        The percentage representation of the input decimal.

    """

    if decimal < 0:

        return 0

    elif decimal > 1:

        return 100

    else:

        return round(decimal * 100)

