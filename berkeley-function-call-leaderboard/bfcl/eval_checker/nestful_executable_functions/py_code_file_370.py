from typing import List



def calculate_normalized_values(numbers: List[float]) -> List[float]:

    """Calculates the normalized values of a list of numbers.

    Args:

        numbers: A list of numbers.

    """

    total_sum = sum(numbers)

    normalized_values = [number / total_sum for number in numbers]

    return normalized_values

