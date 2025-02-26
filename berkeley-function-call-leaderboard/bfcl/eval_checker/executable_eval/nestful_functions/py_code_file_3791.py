from typing import Union



def range_check(val: Union[int, float], min: Union[int, float], max: Union[int, float]) -> Union[int, float]:

    """Checks if a value is within a certain range and returns the value if it is, otherwise returns the boundary value.



    Args:

        val: The value to check.

        min: The minimum value of the range.

        max: The maximum value of the range.

    """

    if min <= val <= max:

        return val

    elif val < min:

        return min

    else:

        return max

