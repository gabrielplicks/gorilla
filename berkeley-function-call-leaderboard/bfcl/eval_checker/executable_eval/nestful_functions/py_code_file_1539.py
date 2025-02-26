from typing import Union



def subtract_or_multiply(x: int, y: int) -> Union[int, int]:

    """

    Returns the result of x minus y if the result is positive, otherwise it returns the result of x multiplied by y.



    Args:

        x: The first integer.

        y: The second integer.

    """

    diff = x - y

    if diff > 0:

        return diff

    else:

        return x * y

