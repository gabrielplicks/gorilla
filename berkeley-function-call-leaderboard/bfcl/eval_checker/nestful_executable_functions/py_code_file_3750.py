from typing import Union



def format_decimal(value: Union[float, int], precision: int) -> str:

    """

    Formats a decimal value with a specific precision. For example,

    `format_decimal(1.2345, 2)` returns "1.23".

    Args:

        value: The decimal value to be formatted.

        precision: The number of digits after the decimal point.

    """

    return f"{value:.{precision}f}"

