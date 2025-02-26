from typing import List, Union



def get_first_negative(numbers: List[Union[int, float]]) -> Union[int, float, None]:

    """Returns the first negative number in the sequence. If no negative numbers are present, the function returns None.



    Args:

        numbers: A sequence of numbers.

    """

    for number in numbers:

        if number < 0:

            return number

    return None

