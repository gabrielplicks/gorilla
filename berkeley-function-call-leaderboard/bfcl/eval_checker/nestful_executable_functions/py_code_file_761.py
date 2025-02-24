from typing import Union



def sum_range(start: Union[int, float], stop: Union[int, float]) -> int:

    """Calculates the sum of all integers in the range from start to stop, inclusive.



    Args:

        start: The starting integer of the range.

        stop: The ending integer of the range.



    Raises:

        TypeError: If start or stop is not an integer.

    """

    if not isinstance(start, int) or not isinstance(stop, int):

        raise TypeError("start and stop must be integers")



    return sum(range(start, stop+1))

