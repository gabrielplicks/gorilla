from typing import Union



def format_number_with_precision(x: float, p: int) -> str:

    """Formats a floating-point number `x` with a given precision `p`.

    If `x` is less than 0.001, it formats it using scientific notation.

    Args:

        x: The floating-point number to format.

        p: The precision to use for formatting.

    """

    if x < 0.001:

        return format(x, f'.{p}e')

    else:

        return format(x, f'.{p}f')

