from typing import List, Union



def get_second_highest(numbers: List[Union[int, float]]) -> Union[int, float, None]:

    """

    Returns the second highest number in a list of numbers.

    If the list has less than two numbers, returns None.



    Args:

        numbers: A list of numbers.

    """

    max_val = -float('inf')

    second_max_val = -float('inf')

    for num in numbers:

        if num > max_val:

            second_max_val = max_val

            max_val = num

        elif num > second_max_val:

            second_max_val = num

    if max_val == -float('inf'):

        return None

    return second_max_val

