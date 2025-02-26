from typing import List



def max_minus_min(numbers: List[float]) -> float:

    """Calculates the difference between the maximum and minimum values of a list of numbers.

    Args:

        numbers: A list of numbers.

    """

    if len(numbers) < 2:

        return 0

    max_val = min_val = numbers[0]

    for num in numbers:

        if num > max_val:

            max_val = num

        elif num < min_val:

            min_val = num

    return max_val - min_val

