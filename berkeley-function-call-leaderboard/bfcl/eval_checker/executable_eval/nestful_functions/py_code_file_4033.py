from typing import List



def find_largest_number(lst: List[int]) -> int:

    """Finds the largest number in a list without using the built-in max function.



    Args:

        lst: A list of integers.



    Returns:

        The largest number in the list.

    """

    largest_number = float('-inf')  # Initialize largest number to negative infinity



    for num in lst:

        if num > largest_number:

            largest_number = num



    return largest_number

