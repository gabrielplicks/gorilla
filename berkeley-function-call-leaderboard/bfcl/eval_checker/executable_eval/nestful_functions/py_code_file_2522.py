from typing import List



def append_even(numbers: List[int]) -> List[int]:

    """Appends even numbers from a list to a new list.



    Args:

        numbers: A list of integers.



    Returns:

        A new list of even integers.

    """

    even_numbers = []



    for num in numbers:

        if num % 2 == 0:

            even_numbers.append(num)



    return even_numbers

