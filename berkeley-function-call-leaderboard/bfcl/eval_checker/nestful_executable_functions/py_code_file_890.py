from typing import List



def filter_positive_values_or_negative_infinity(numbers: List[float]) -> List[float]:

    """Filters a list of numbers and returns a new list with only the positive values in the input list.

    If a negative value is encountered, return -inf as the value instead.

    Args:

        numbers: The input list of numbers.

    """

    return [float('-inf') if number < 0 else number for number in numbers]

