from typing import Tuple



def four_operations(x: float, y: float) -> Tuple[float, float, float, float]:

    """Calculates the sum, difference, product, and quotient of two numbers.



    Args:

        x: The first number.

        y: The second number.



    Returns:

        A tuple containing the sum, difference, product, and quotient of the two numbers.

    """

    q, r = divmod(x, y)

    return x + y, x - y, q * y + r, q

