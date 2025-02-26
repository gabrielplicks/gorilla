from typing import List



def sum_even_integers(integer_list: List[int]) -> int:

    """Calculates the sum of all even integers in a given list.

    Args:

        integer_list: A list of integers.

    """

    sum_even_numbers = 0

    for number in integer_list:

        if number % 2 == 0:  # Check if the number is even

            sum_even_numbers += number  # Add the number to the sum

    return sum_even_numbers

