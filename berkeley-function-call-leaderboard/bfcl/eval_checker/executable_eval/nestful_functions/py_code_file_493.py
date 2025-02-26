from functools import reduce

from typing import List



def exclude_product(numbers: List[int]) -> List[int]:

    """Calculates the product of all numbers in a list, excluding the number at each index.



    Args:

        numbers: A list of integers.



    Returns:

        A new list where each element is the product of all numbers in the original list except the one at that index.

    """

    return [reduce(lambda x, y: x * y, [num for num in numbers if num != n]) for n in numbers]

