from typing import Union



def square_with_power(x: int, power: Union[int, float] = 2) -> float:

    """Calculates the power of a number.



    Args:

        x: The number to be powered.

        power: The power to which the number is to be raised. Default is 2.

    """

    if power == 2:

        return x * x

    return x ** power

