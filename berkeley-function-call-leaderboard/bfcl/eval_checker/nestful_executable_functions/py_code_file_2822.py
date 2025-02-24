from typing import List



def mean_of_squares(numbers: List[float]) -> float:

    """Calculates the mean of the squares of a list of numbers.



    Args:

        numbers: A list of numbers.



    Returns:

        The mean of the squares of the numbers in the list.

    """

    if not numbers:

        return 0



    squares_sum = sum(number * number for number in numbers)

    mean = squares_sum / len(numbers)



    return mean

