from typing import Tuple



def mutate_point(x: float, y: float, dx: float, dy: float) -> Tuple[float, float]:

    """Mutates the given point (x, y) by adding x to dx, and y to dy.

    Returns the result as a tuple (x', y').

    Applies the mutation only if the sum of x and dx and the sum of y and dy are both positive numbers.

    Args:

        x: The x-coordinate of the point.

        y: The y-coordinate of the point.

        dx: The amount to add to x.

        dy: The amount to add to y.

    """

    if x + dx > 0 and y + dy > 0:

        return (x + dx, y + dy)

    else:

        return (x, y)

