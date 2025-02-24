from typing import Union



def decimal_of_power_of_ten(n: int) -> int:

    """Calculates the decimal value of the binary number 10^n.



    Args:

        n: The exponent of the binary number 10.



    Returns:

        The decimal value of the binary number 10^n.

    """

    if n == 0:

        return 1

    else:

        return 10 * decimal_of_power_of_ten(n - 1)

