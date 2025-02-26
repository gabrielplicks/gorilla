from typing import List



def largest_sum(numbers: List[int]) -> int:

    """Calculates the largest sum of any two numbers in a list of non-negative integers.



    Args:

        numbers: A list of non-negative integers.



    Returns:

        The largest sum of any two numbers in the list.

    """

    numbers.sort(reverse=True)

    return numbers[0] + numbers[1]

