from typing import Union



def sqrt(x: Union[int, float], n: int = 2) -> float:

    """Calculates the square root of a given number `x` to `n` decimal places.

    Args:

        x: The number to calculate the square root of.

        n: The number of decimal places to round the result to.

    """

    guess = x / 2

    while True:

        new_guess = (guess + x / guess) / 2

        if abs(new_guess - guess) < 10 ** -n:

            break

        guess = new_guess

    return round(guess, n)

