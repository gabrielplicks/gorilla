from typing import List



def sum_odd_numbers(numbers: List[int]) -> int:

    """Calculates the sum of all odd numbers in a list.



    Args:

        numbers: A list of numbers.



    Returns:

        The sum of all odd numbers in the list.

    """

    sum_odd = 0

    for num in numbers:

        if num % 2 != 0:

            sum_odd += num

    return sum_odd

