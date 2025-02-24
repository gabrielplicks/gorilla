from typing import List



def median_without_sorting(numbers: List[float]) -> float:

    """Calculates the median of a given list of numbers without sorting them.

    If the length of the list is even, the median should be the average of the two middle numbers.

    Otherwise, it is the middle number.

    Args:

        numbers: A list of numbers.

    """

    length = len(numbers)

    if length % 2 == 0:

        return (numbers[length // 2 - 1] + numbers[length // 2]) / 2

    else:

        return numbers[length // 2]

