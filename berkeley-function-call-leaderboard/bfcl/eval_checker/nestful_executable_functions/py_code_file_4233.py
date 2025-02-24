from typing import List, Tuple



def split_integers(numbers: List[int]) -> Tuple[List[int], List[int]]:

    """Splits a list of integers into two sorted groups based on whether the numbers are even or odd.



    Args:

        numbers: A list of integers.



    Returns:

        A tuple containing two lists. The first list contains the even numbers, and the second list contains the odd numbers.

    """

    sorted_numbers = sorted(numbers, key=lambda x: x % 2)

    even_numbers = [num for num in sorted_numbers if num % 2 == 0]

    odd_numbers = [num for num in sorted_numbers if num % 2 == 1]

    return even_numbers, odd_numbers

