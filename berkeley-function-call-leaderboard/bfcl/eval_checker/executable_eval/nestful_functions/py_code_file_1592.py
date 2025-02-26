from typing import List



def separate_even_and_odd(numbers: List[int]) -> (List[int], List[int]):

    """Separates a list of numbers into two lists: one containing only the even numbers and the other containing only the odd numbers.



    Args:

        numbers: A list of integers.



    Returns:

        A tuple containing two lists: the first list contains the even numbers and the second list contains the odd numbers.

    """

    even_numbers = [number for number in numbers if number % 2 == 0]

    odd_numbers = [number for number in numbers if number % 2 == 1]

    return even_numbers, odd_numbers

