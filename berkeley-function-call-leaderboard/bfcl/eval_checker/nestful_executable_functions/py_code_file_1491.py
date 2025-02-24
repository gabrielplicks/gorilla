from typing import List, Optional



def first_even_greater_than_3(numbers: List[int]) -> Optional[int]:

    """Finds the first even number that is greater than 3 in a list of integers.



    Args:

        numbers: A list of integers.



    Returns:

        The first even number that is greater than 3, or None if no such number exists.

    """

    filtered_numbers = [

        number

        for number in numbers

        if number % 2 == 0 and number > 3

    ]

    return next(iter(filtered_numbers), None)

