from typing import List



def calculate_significance_level(values: List[float], p_value: float) -> float:

    """Calculates the significance level based on the given list of values and a p-value threshold.



    Args:

        values: A list of values.

        p_value: The p-value threshold.



    Returns:

        The significance level.

    """

    if not values:

        return 0



    significance_level = 1 - p_value**(1 / len(values))

    return significance_level

