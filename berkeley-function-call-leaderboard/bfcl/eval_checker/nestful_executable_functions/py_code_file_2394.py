from typing import Union



def fx(x: Union[int, float]) -> float:

    """Evaluates the mathematical formula: fx(x) = 2x^3 - 3x^2 - 32x + 1.



    Args:

        x: The input value.

    """

    return 2 * x**3 - 3 * x**2 - 32 * x + 1

