from typing import Union



def convert_scientific_notation(s: str) -> float:

    """Converts a scientific notation string to a float value.



    Args:

        s: A string representing a scientific notation number.



    Returns:

        A float value equivalent to the scientific notation string.

    """

    x, y = s.split("E")

    mantissa = float(x)

    exponent = int(y)

    if exponent < 0:

        return mantissa / 10**abs(exponent)

    else:

        return mantissa * 10**exponent

