from itertools import filterfalse

from typing import List



def filter_lisp_style(numbers: List[int]) -> List[int]:

    """Filters an input list of numbers using a LISP-style list comprehension.

    Given a list of numbers, the function returns a list containing only the odd numbers.

    Uses a list comprehension to iterate over the input list and a condition to filter only the odd numbers.



    Args:

        numbers: A list of integers.



    Returns:

        A list of integers containing only the odd numbers from the input list.

    """

    return list(filterfalse(lambda x: x % 2 == 0, numbers))

