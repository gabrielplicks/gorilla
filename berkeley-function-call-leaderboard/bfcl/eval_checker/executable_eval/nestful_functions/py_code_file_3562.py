from typing import List, Tuple



def filter_odd_even(numbers: List[int]) -> Tuple[List[int], List[int]]:

    """Filters a list of integers into two lists, one with the odd numbers and one with the even numbers.



    Args:

        numbers: The list of integers to filter.



    Returns:

        A tuple of two lists, with the first element being the list of odd numbers and the second element being the list of even numbers.

    """

    odd_numbers = []

    even_numbers = []



    for num in numbers:

        if num % 2 == 0:

            even_numbers.append(num)

        else:

            odd_numbers.append(num)



    return odd_numbers, even_numbers

