from typing import Union



def abs_limited_to(x: int, min_value: Union[int, float]) -> Union[int, float]:

    """Calculates the absolute value of an integer `x`, limited to a minimum value of `min_value`.



    Args:

        x: The integer to calculate the absolute value of.

        min_value: The minimum value to limit the absolute value to.



    Returns:

        The absolute value of `x`, limited to a minimum value of `min_value`.

    """

    return min(abs(x), min_value)

