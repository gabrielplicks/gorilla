from typing import List



def percent_positive(numbers: List[int]) -> float:

    """Calculates the percentage of positive integers in a list.



    Args:

        numbers: A list of positive integers.



    Returns:

        The percentage of positive integers in the list.

    """

    positive_count = 0

    for num in numbers:

        if num > 0:

            positive_count += 1

    percentage = positive_count / len(numbers) * 100

    return percentage

