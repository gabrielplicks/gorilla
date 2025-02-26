from typing import List



def sum_of_positive_integers(numbers: List[int]) -> int:

    """Computes the sum of all positive integers in a given list.



    Args:

        numbers: A list of positive integers.



    Returns:

        The sum of all positive integers in the list.

    """

    sum_of_positive = 0

    for num in numbers:

        if num > 0:

            sum_of_positive += num

    return sum_of_positive

