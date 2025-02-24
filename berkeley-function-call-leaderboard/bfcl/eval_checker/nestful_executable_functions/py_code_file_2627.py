import numpy as np



zero = 0

one = 1

two = 2

three = 3

four = 4

five = 5

six = 6

seven = 7

eight = 8

nine = 9

ten = 10

twenty = 20

thirty = 30

sixty = 60

hundred = 100

thousand = 10**3

million = 10**6

billion = 10**9



def pi() -> float:

    """Calculates the value of the mathematical constant pi.



    Args:

        None



    Returns:

        The value of pi.

    """

    return 4 * (4 * np.arctan(1 / 5) - np.arctan(1 / 239))

