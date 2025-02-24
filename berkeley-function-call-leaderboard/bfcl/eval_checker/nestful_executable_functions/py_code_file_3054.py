from typing import Tuple



def beta_value(start: int, end: int) -> float:

    """Calculates the beta value for the specified range.

    Args:

        start: The start of the range.

        end: The end of the range.

    """

    beta = (end - start) * (end - start + 1) / 2

    return float(beta)

