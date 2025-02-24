from typing import List



def calculate_95th_percentile(numbers: List[float]) -> float:

    """Calculates the 95th percentile value of a list of numbers.



    Args:

        numbers: A list of numbers.



    Returns:

        The 95th percentile value of the list.

    """

    sorted_numbers = sorted(numbers)

    index = int(len(numbers) * 0.95)

    return sorted_numbers[index]

