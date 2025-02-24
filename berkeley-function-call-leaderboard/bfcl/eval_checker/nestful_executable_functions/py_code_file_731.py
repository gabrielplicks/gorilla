from typing import Union



def format_significant(num: Union[int, float], n: int) -> str:

    """Formats a number `num` with a specified number of significant figures `n` and returns a string.

    Args:

        num: The number to be formatted.

        n: The number of significant figures.

    """

    return f"{num:.{n}g}"

