from typing import Union



def handle_zero_division(x: float, y: float) -> Union[float, str]:

    """Calculates the quotient of two numbers, handling the case when the denominator is zero.



    Args:

        x: The numerator.

        y: The denominator.



    Returns:

        The quotient of x and y, or a special value if y is zero.

    """

    if y == 0:

        return 'Error: cannot divide by zero.'

    else:

        return x / y

