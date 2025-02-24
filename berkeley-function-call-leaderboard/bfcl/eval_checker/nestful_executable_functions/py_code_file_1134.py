from typing import Union



def find_closest_root(n: int) -> Union[int, float]:

    """Finds the closest integer root of a given integer.



    Args:

        n: The integer to find the closest root of.



    Returns:

        The closest integer root of `n`.

    """

    if n == 0:

        return 0

    root = int(n ** 0.5)

    if root ** 2 == n:

        return root

    guess = n / 2

    while abs(guess * guess - n) > 0.001:

        guess = (guess + n / guess) / 2

    closest_root = int(round(guess))



    return closest_root

