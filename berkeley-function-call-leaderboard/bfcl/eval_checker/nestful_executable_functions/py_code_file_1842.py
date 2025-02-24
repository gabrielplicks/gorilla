from typing import Callable



def f_composed(x: float) -> float:

    """

    Returns the composition of the following functions: f1(x) = x^2 + 1, f2(x) = x/2, and f3(x) = x + 1.

    Args:

        x: The input value.

    """

    def f1(x: float) -> float:

        return x**2 + 1



    def f2(x: float) -> float:

        return x / 2



    def f3(x: float) -> float:

        return x + 1



    return f3(f2(f1(x)))

