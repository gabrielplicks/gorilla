from typing import List



def swap_max_min(numbers: List[int]) -> List[int]:

    """Swaps the maximum and minimum values in a list of numbers.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of numbers with the maximum and minimum values swapped.

    """

    max_value = max(numbers)

    min_value = min(numbers)



    max_index = numbers.index(max_value)

    min_index = numbers.index(min_value)



    numbers.pop(max_index)

    numbers.insert(max_index, min_value)



    numbers.pop(min_index)

    numbers.insert(min_index, max_value)



    return numbers

