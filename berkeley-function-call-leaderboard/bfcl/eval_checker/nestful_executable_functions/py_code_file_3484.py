from typing import Union



def recursive_function(n: Union[int, float]) -> Union[int, float]:

    """Calculates the value of the recursive equation.

    Args:

        n: The input value.

    """

    if n == 0:

        return 1

    else:

        return n * recursive_function(n - 1)

