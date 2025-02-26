from typing import Union



def min_of_two_numbers(x: Union[int, float, None], y: Union[int, float, None]) -> Union[int, float, None]:

    """Returns the smaller non-negative number from two given numbers.

    If either number is negative, the function returns None.

    If both numbers are undefined, the function also returns None.



    Args:

        x: The first number.

        y: The second number.

    """

    if x is None and y is None:

        return None

    if x is None:

        return y

    elif y is None:

        return x

    if x < 0 or y < 0:

        return None

    return min(x, y)

