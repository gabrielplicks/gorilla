from typing import Union



def int2float(x: int) -> float:

    """Converts an integer to a floating-point number.



    Args:

        x: The integer to convert.



    Returns:

        The floating-point representation of the integer.

    """

    result = 0

    is_negative = x < 0

    x = abs(x)

    power = 1



    while x > 0:

        digit = x % 10

        result += digit * power

        power *= 10

        x //= 10



    if is_negative:

        result = -result



    return float(result)

