from typing import List



def calculate_average(numbers: List[float]) -> str:

    """Calculates the average of a list of numbers and returns the result as a string.



    Args:

        numbers: A list of numbers.



    Returns:

        The average of the numbers, formatted as a string with two decimal places.

    """

    average = sum(numbers) / len(numbers)

    return f"{average:.2f}"

