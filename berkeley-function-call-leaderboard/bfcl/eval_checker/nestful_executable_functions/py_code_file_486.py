from typing import List, Union



def sum_integers(integer_list: List[Union[int, str]]) -> int:

    """Calculates the sum of all integers in a given list.

    If a value is a string representation of a number, it is converted into an integer before summing.

    If the list includes a value that is neither an integer nor a string representation of a number,

    a ValueError is raised.

    Args:

        integer_list: A list of integers and/or string representations of numbers.

    """

    total = 0



    for value in integer_list:

        try:

            total += int(value)

        except ValueError:

            raise ValueError("A value in the list is not an integer or a string representation of a number.")



    return total

