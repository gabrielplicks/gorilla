from typing import Tuple



def sort_descending(x: float, y: float) -> Tuple[float, float]:

    """Sorts two variables in descending order using conditional swapping and ternary operators.



    Args:

        x: The first variable.

        y: The second variable.



    Returns:

        A tuple of the variables in descending order.

    """

    if x == y:

        return (x, y)

    else:

        greater = x if x > y else y

        smaller = x if x < y else y

        return (greater, smaller)

