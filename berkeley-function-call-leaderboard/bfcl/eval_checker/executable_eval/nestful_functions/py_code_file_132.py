from typing import List, Tuple



def sum_and_avg(numbers: List[float]) -> Tuple[float, int]:

    """Calculates the sum and average of a list of three numbers.

    Args:

        numbers: A list of three numbers.

    Returns:

        A tuple containing the sum and average of the input numbers.

    """

    total_sum = sum(numbers)

    length = len(numbers)

    average = round(total_sum / length)

    return (total_sum, average)

