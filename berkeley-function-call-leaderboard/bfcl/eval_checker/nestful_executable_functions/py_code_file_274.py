from typing import List



def calculate_average_except_zeros_and_negatives(numbers: List[float]) -> float:

    """

    Calculates the average of a list of numbers, excluding zeroes and negative numbers.

    The function raises a ValueError if the list is empty.

    Args:

        numbers: A list of numbers.

    """

    if not numbers:

        raise ValueError("Cannot calculate average of an empty list.")



    filtered_nums = [num for num in numbers if num > 0]

    total = sum(filtered_nums)

    count = len(filtered_nums)



    if total == 0 or count == 0:

        return 0



    return total / count

