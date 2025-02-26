import math



def calculate_inverse_log(log_val: float, base: float) -> float:

    """Calculates the inverse logarithm value given a logarithm value and its base.



    Args:

        log_val: The logarithm value.

        base: The base of the logarithm.



    Returns:

        The inverse logarithm value.

    """

    return base ** (log_val * (2 / math.log(base, 2)))

