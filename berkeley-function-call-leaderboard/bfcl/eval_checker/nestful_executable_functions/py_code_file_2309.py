from typing import Union



def min_max_inverse(x: Union[int, float]) -> float:

    """Calculates the min-max inverse of a floating-point number `x`.



    Args:

        x: The input floating-point number.



    Returns:

        The min-max inverse of `x`, which is a number in the range `[0, 1]`.

    """

    y = (x - 0) / (1 - 0)  # Apply min-max normalization formula with min(x) = 0 and max(x) = 1

    return y

