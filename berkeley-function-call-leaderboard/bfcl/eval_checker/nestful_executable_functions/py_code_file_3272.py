from typing import Union



def check_divisible_by_3(num: Union[int, float]) -> bool:

    """Checks whether a given number is divisible by 3.



    Args:

        num: The number to check.



    Returns:

        True if the number is divisible by 3, False otherwise.

    """

    return num % 3 == 0

