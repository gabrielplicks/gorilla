from typing import List



def iterate_over_numbers(numbers: List[float]) -> List[float]:

    """

    Takes a list of numbers and returns a new list containing the square of each number.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of squared numbers.

    """

    squared_values = []

    for number in numbers:

        squared_value = number ** 2

        squared_values.append(squared_value)

    return squared_values

