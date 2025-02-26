def add_1_abs_value(x: int) -> int:

    """Calculates the absolute value of an integer plus one without using the built-in abs() function.



    Args:

        x: The integer to calculate the absolute value of.



    Returns:

        The absolute value of the integer plus one.

    """

    if x >= 0:

        return x + 1

    return -x + 1

