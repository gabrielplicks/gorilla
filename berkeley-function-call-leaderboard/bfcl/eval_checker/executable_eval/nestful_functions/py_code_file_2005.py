from typing import List



def sum_cubes(numbers: List[int]) -> int:

    """Computes the sum of cubes of all the numbers in a list.



    Args:

        numbers: A list of integers.



    Returns:

        The sum of the cubes of the numbers in the list.

    """

    return sum(x ** 3 for x in numbers)

