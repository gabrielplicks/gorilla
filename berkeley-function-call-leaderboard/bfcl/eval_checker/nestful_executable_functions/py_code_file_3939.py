from typing import Union



def percentage_to_decimal(percentage: Union[int, float]) -> float:

    """Converts a percentage to a decimal.



    Args:

        percentage: A float or integer representing a percentage.



    Returns:

        A float representing the value in decimal form.

    """

    return float(percentage) / 100.0

