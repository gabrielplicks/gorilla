from typing import Union



def y_calc(x: Union[int, float]) -> Union[int, float]:

    """Calculates the value of y based on the given mathematical formula.

    Args:

        x: The input value.

    """

    if x < 0:

        return x

    return abs(x)

