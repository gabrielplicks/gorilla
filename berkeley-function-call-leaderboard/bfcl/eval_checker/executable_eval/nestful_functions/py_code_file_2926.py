from typing import Union



def add_with_rounding(a: Union[int, float], b: Union[int, float], rounding: bool) -> int:

    """Adds two numbers with or without rounding, depending on the value of the `rounding` argument.



    Args:

        a: The first number to add.

        b: The second number to add.

        rounding: If `True`, the result will be rounded to the nearest integer.



    Returns:

        The result of the addition as an integer.

    """

    result = a + b

    if rounding:

        result = round(result)

    return int(result)

