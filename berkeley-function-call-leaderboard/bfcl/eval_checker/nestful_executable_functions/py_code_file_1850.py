from typing import Tuple



def polynomial_expression(x: float, a: float, b: float, c: float) -> float:

    """Calculates the value of the polynomial expression given the coefficients and the value of x.



    Args:

        x: The value of the variable.

        a: The coefficient of the quadratic term.

        b: The coefficient of the linear term.

        c: The constant term.



    Returns:

        The value of the polynomial expression.

    """

    return a * x ** 2 + b * x + c

