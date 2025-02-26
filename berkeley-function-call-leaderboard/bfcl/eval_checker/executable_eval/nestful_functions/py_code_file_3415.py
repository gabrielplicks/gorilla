from typing import List



def num_evens_list_comprehension(numbers: List[int]) -> int:

    """Calculates the number of even numbers in a list using list comprehension.

    Args:

        numbers: A list of integers.

    """

    return len([num for num in numbers if num % 2 == 0])



def num_evens_filter(numbers: List[int]) -> int:

    """Calculates the number of even numbers in a list using the filter function.

    Args:

        numbers: A list of integers.

    """

    even_numbers = filter(lambda num: num % 2 == 0, numbers)

    return len(list(even_numbers))

