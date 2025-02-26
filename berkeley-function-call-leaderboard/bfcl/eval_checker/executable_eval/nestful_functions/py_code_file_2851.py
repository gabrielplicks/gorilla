from typing import Union



def check_divisibility(x: Union[int, float]) -> str:

    """Checks if the given integer is divisible by 3 and returns the corresponding message.



    Args:

        x: The integer to check for divisibility.



    Returns:

        A string indicating whether the given integer is divisible by 3 or not.

    """

    if x % 3 == 0:

        return f"{x} is a good number"

    else:

        return f"{x} is not a good number"

