from typing import Union



def check_special(x: Union[float, int]) -> bool:

    """Checks if a numerical value is "special" (i.e., it is 1/3, 1/5, or 1/7).



    Args:

        x: The numerical value to check.



    Returns:

        True if the value is "special", False otherwise.

    """

    return x in (1/3, 1/5, 1/7)

