from typing import List



def get_mean(numbers: List[float]) -> float:

    """Calculates the mean of a list of numbers.



    Args:

        numbers: A list of numbers.

    """

    total = 0

    for num in numbers:

        total += num



    return total / len(numbers)

