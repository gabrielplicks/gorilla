from typing import List



def is_all_even(numbers: List[int]) -> bool:

    """

    Returns `True` if all elements in `numbers` are even, and `False` otherwise.



    Args:

        numbers: A list of integers.



    Returns:

        `True` if all elements in `numbers` are even, and `False` otherwise.



    Examples:

        >>> is_all_even([2, 4, 6])

        True

        >>> is_all_even([1, 2, 3, 4])

        False

        >>> is_all_even([])

        True

        >>> is_all_even([3, 5, 7])

        False

    """

    return all(number % 2 == 0 for number in numbers)

