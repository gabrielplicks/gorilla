from typing import Union



def is_multiple_of_3_or_5(n: Union[int, float]) -> bool:

    """Checks if a number is a multiple of 3 or 5.



    Args:

        n: The number to check.



    Returns:

        True if the number is a multiple of 3 or 5, False otherwise.

    """

    return (n % 3 == 0) or (n % 5 == 0)

