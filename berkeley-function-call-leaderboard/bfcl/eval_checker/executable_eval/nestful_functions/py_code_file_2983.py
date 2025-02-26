from typing import Union



def Hash(x: Union[int, float]) -> int:

    """Calculates the hash of a given integer `x`.



    The hash is calculated by taking the modulo of `x` with 100.



    Args:

        x: The integer to calculate the hash of.



    Returns:

        The hash of `x`.

    """

    return int(x % 100)

