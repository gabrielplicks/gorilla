from typing import List



def apply_functions(numbers: List[int]) -> int:

    """Applies a sequence of functions to a list of integers.



    The sequence of functions is as follows:

    1. Filter out the even numbers

    2. Sort the remaining numbers in descending order

    3. Multiply each number by 2

    4. Calculate the sum of the resulting list



    Args:

        numbers: A list of integers.



    Returns:

        The sum of the resulting list after applying the sequence of functions.

    """

    return sum_list(multiply_by_2(descending_sort(even_filter(numbers))))

