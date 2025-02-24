import numpy as np



def exponential_model(a: float, b: float, c: float, x: float) -> float:

    """Calculates the value of the exponential model function at a given point.



    Args:

        a: The coefficient of the exponential term.

        b: The exponent of the exponential term.

        c: The constant term.

        x: The point at which to evaluate the function.

    """

    return a * np.exp(b * x) + c

