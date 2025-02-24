from typing import Union



def max_abs(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:

    """Computes the maximum absolute value of two numbers, without using the `abs` function.



    Args:

        x: The first number.

        y: The second number.

    """

    abs_x = x if x >= 0 else -x

    abs_y = y if y >= 0 else -y

    if abs_x > abs_y:

        return abs_x

    else:

        return abs_y

