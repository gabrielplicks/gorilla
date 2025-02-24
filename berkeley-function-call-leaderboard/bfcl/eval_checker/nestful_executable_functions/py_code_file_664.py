from typing import List



def average_string(numbers: List[float]) -> str:

    """Calculates the average of a list of floating-point numbers and returns it as a string with exactly two decimal places.



    Args:

        numbers: A list of floating-point numbers.



    Returns:

        The average of the numbers as a string with exactly two decimal places.

    """

    total = sum(numbers)

    count = len(numbers)

    average = total / count

    return f"{average:.2f}"

