from typing import List, Union



def closest_number(numbers: List[Union[int, float]], x: Union[int, float]) -> Union[int, float]:

    """

    Returns the number in the list that is closest to x.

    If there are multiple such numbers, returns the one with the smallest absolute difference.



    Args:

        numbers: A list of numbers.

        x: A number.

    """

    closest = None

    min_difference = None

    for number in numbers:

        difference = abs(number - x)

        if min_difference is None or difference < min_difference:

            min_difference = difference

            closest = number

    return closest

