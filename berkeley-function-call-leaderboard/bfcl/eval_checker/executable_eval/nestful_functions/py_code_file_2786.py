from typing import Union



def piecewise_linear(x: Union[int, float]) -> Union[int, float]:

    """Calculates the piecewise linear function of the variable x.



    Args:

        x: The input variable.



    Returns:

        The value of the piecewise linear function at x.

    """

    if x < 0:

        return 0

    elif x < 1:

        return x

    else:

        return 1

