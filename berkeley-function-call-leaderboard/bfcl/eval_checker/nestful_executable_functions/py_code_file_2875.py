from typing import List



def rearrange_list(numbers: List[int]) -> List[int]:

    """Rearranges a list of numbers so that all even numbers appear first, followed by all odd numbers.



    Args:

        numbers: A list of integers.



    Returns:

        A new list with the same numbers, but with all even numbers appearing first, followed by all odd numbers.

    """

    return [number for number in numbers if number % 2 == 0] + [number for number in numbers if number % 2 == 1]

