from typing import Union



def is_narcissistic(n: Union[int, float]) -> bool:

    """Checks if a positive integer n is a narcissistic number.



    A narcissistic number is a number that equals the sum of its own digits in the nth power.

    For example, 153 is a narcissistic number since 153 = 1^3 + 5^3 + 3^3.



    Args:

        n: The positive integer to check.



    Returns:

        True if n is a narcissistic number, and False otherwise.

    """

    if not isinstance(n, int) or n <= 0:

        raise ValueError("Input must be a positive integer.")



    digits = str(n)

    digit_count = len(digits)

    return sum(int(digit) ** digit_count for digit in digits) == n

