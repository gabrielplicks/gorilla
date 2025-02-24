from typing import List



def avg(numbers: List[float]) -> float:

    """Calculates the arithmetic mean of a given list of numbers. If the list is empty, return 0.



    Args:

        numbers: A list of numbers.

    """

    if len(numbers) == 0:

        return 0

    return sum(numbers) / len(numbers)

