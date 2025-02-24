def sum_func(x: float, y: float) -> float:

    """Calculates the sum of two numbers `x` and `y`, ensuring the result is at least `x + y` in value and is 0 when `x + y` is less than 1.



    Args:

        x: The first number.

        y: The second number.



    Returns:

        The sum of the two numbers, ensuring the result is at least `x + y` in value and is 0 when `x + y` is less than 1.

    """

    if x == 0 or y == 0:

        return max(x, y)

    result = x + y

    if result < x + y:

        result = x + y

    return result

