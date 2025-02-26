from typing import List, Union



def validate_and_sum(numbers: List[Union[int, float, str]]) -> float:

    """Calculates the sum of only the numbers within the list that are of type int or float.

    Args:

        numbers: A list of numbers that can be of type int, float, or any other type.

    """

    total = 0

    for num in numbers:

        if isinstance(num, (int, float)):

            total += num

    return total

