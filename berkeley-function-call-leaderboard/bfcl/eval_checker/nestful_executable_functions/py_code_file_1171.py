from typing import Optional



def def_func(a: Optional[int] = 1, b: Optional[int] = 2) -> int:

    """Calculates the sum of two positive integers.

    If the first argument is missing, it is set to `1`.

    If the second argument is missing, it is set to `2`.

    Args:

        a: The first positive integer. Defaults to `1`.

        b: The second positive integer. Defaults to `2`.

    """

    return a + b

