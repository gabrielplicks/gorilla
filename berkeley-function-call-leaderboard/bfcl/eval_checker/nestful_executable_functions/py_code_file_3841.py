from typing import Union



def power_of_scalar(base: Union[int, float], power: int) -> Union[int, float]:

    """Calculates the power of a scalar value.



    Args:

        base: A scalar value (integer or float).

        power: A positive integer.



    Raises:

        AssertionError: If the input types are invalid.

    """

    assert isinstance(base, (int, float)) and isinstance(power, int) and power >= 0, "Invalid input types"

    return base ** power

