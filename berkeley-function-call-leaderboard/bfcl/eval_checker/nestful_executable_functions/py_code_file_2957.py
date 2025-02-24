import math



def sqrt_rounded(number: float) -> int:

    """Calculates the square root of a number, rounded to the nearest integer.

    Args:

        number: The number to calculate the square root of.

    Raises:

        Exception: If the input is not a positive number.

    """

    if number < 0:

        raise Exception('Input must be a positive number.')



    square_root = math.sqrt(number)

    rounded_root = round(square_root)



    return rounded_root

