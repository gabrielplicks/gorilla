from typing import Union



def difference_calculator(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:

    """Calculates the difference between two numbers `x` and `y`.



    Args:

        x: The first number.

        y: The second number.



    Raises:

        Exception: If `x` and `y` are not integer or float values.

    """

    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):

        raise Exception("x and y must be integer or float values")



    return x - y

