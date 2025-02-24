from typing import Tuple



def calculate_ratio_and_compare(numerator: float, denominator: float) -> Tuple[float, bool]:

    """Calculates the ratio of two numbers and checks if the numerator is greater than the denominator.



    Args:

        numerator: The numerator value.

        denominator: The denominator value.



    Returns:

        A tuple containing the ratio of the values and a boolean indicating whether `numerator` is greater than `denominator`.

    """

    ratio = numerator / denominator

    is_numerator_greater = numerator > denominator

    return ratio, is_numerator_greater

