from typing import List



def sum_of_absolute_values(array: List[int]) -> int:

    """Calculates the sum of the absolute values of the elements in an array.



    Args:

        array: A list of integers.



    Returns:

        The sum of the absolute values of the elements in the array.

    """

    return sum(abs(x) for x in array)

