from typing import Union



def max_number(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:

    """

    Returns the larger of the two numbers `x` and `y`. If the two numbers are equal, returns `x` as the larger number.



    Args:

        x: The first number.

        y: The second number.

    """

    if x > y:

        return x

    elif x < y:

        return y

    else:

        return x

