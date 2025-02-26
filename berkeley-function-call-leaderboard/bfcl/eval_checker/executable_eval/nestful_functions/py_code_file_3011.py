from typing import List



def add_1(numbers: List[int]) -> List[int]:

    """Adds 1 to each number in the list.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of numbers with 1 added to each number.

    """

    new_numbers = []

    for number in numbers:

        try:

            new_number = number + 1

            new_numbers.append(new_number)

        except TypeError:

            new_numbers.append(number)

    return new_numbers

