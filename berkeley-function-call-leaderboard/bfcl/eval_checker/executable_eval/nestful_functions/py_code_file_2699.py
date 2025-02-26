from typing import Union



def get_power_of_two(n: Union[int, float]) -> int:

    """Returns the power of two that is nearest to the provided input `n`.



    Args:

        n: The input number.



    Returns:

        The power of two that is nearest to `n`.

    """

    if n <= 0:

        return 0



    i = 1

    while 2 ** i < n:

        i += 1



    return 2 ** (i - 1)

