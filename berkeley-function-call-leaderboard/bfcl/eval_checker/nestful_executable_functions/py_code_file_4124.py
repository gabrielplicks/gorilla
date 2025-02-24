from typing import List



def reorder_numbers(nums: List[int]) -> List[int]:

    """Reorders a list of numbers in such a way that all even numbers appear before all odd numbers.



    Args:

        nums: A list of numbers.



    Returns:

        A new list with all even numbers appearing before all odd numbers.

    """

    return [num for num in nums if num % 2 == 0] + [num for num in nums if num % 2 == 1]

