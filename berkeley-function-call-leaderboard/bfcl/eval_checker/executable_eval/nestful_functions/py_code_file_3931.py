from typing import Tuple



def get_simplified_fraction(numerator: int, denominator: int) -> str:

    """Simplifies a fraction and returns it as a string in the format "numerator/denominator".



    Args:

        numerator: The numerator of the fraction.

        denominator: The denominator of the fraction.



    Returns:

        The simplified fraction as a string in the format "numerator/denominator".

    """

    divisor = min(abs(numerator), abs(denominator))

    while divisor > 1:

        if numerator % divisor == 0 and denominator % divisor == 0:

            break

        divisor -= 1

    numerator //= divisor

    denominator //= divisor

    sign = '-' if numerator * denominator < 0 else ''



    return f'{sign}{abs(numerator)}/{abs(denominator)}'

