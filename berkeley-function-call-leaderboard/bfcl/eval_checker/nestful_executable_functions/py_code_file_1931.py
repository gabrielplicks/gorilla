from typing import Union



def is_even_and_more_than_zero(num: int) -> bool:

    """Checks if the argument is even and greater than zero.

    Args:

        num: The integer to be checked.

    """

    return num % 2 == 0 and num > 0



def is_divisible_by_10(num: int) -> bool:

    """Checks if the argument is divisible by 10.

    Args:

        num: The integer to be checked.

    """

    return num % 10 == 0



def is_even_and_more_than_zero_and_divisible_by_10(num: Union[int, float]) -> bool:

    """Checks if an integer is even, greater than zero, and divisible by 10.

    Args:

        num: The integer to be checked.

    """

    return is_even_and_more_than_zero(num) and is_divisible_by_10(num)

