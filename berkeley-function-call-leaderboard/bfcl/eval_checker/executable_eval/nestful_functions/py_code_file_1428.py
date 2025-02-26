def safe_div(numerator: float, denominator: float) -> float:

    """Calculates the result of the equation f(x), where x is the input numerator divided by the input denominator.

    Args:

        numerator: The numerator of the division.

        denominator: The denominator of the division.

    """

    if denominator != 0:

        return numerator / denominator

    else:

        return 0

