from typing import Union



def f(x: Union[int, float]) -> Union[int, float]:

    """

    Returns the value of the expression x^2 + 1 if x < 3 and 0 otherwise.

    Args:

        x: The input value.

    """

    return (x ** 2 + 1) if x < 3 else 0

