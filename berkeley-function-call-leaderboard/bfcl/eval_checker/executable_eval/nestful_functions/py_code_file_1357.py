import math



log10_of_2 = math.log10(2)



def log_sum_product(x: int, y: int) -> float:

    """Calculates the sum of the base-10 logarithms of two integers.



    Args:

        x: The first integer.

        y: The second integer.



    Returns:

        The sum of the base-10 logarithms of `x` and `y`.

    """

    if x <= 0 or y <= 0:

        return None

    log10_x = math.log10(x)

    log10_y = math.log10(y)

    log10_sum = log10_x + log10_y

    return log10_sum

