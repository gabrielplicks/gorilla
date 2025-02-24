from typing import Union



def f_of_x(x: Union[int, float] = 0) -> Union[int, float]:

    """Evaluates the expression f(x) = x^2 + 2*x - 1 and returns its value.



    Args:

        x: The value to evaluate f(x) for.



    Returns:

        The value of f(x) for the given x.

    """

    f = lambda x: x**2 + 2*x - 1

    return f(x)

