from typing import List



def sum_of_negative_numbers(array: List[int]) -> int:

    """Calculates the sum of all negative numbers in the given array.



    Args:

        array: A list of integers.



    Returns:

        The sum of all negative numbers in the array.

    """

    if not array:

        return 0

    sum_of_negatives = 0

    for n in array:

        if n < 0:

            sum_of_negatives += n

    return sum_of_negatives

