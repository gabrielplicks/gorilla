import random

random.seed(42)
import typing



def fluctuate_position(x: float, delta: float) -> float:

    """Generates a random number `r` between `-delta` and `delta`, and then returns `x + r`.



    Args:

        x: The horizontal coordinate of a 2D point.

        delta: The maximum allowable fluctuation.

    """

    r = random.uniform(-delta, delta)

    return x + r

