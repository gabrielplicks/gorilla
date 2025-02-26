import random

random.seed(42)
from typing import Union



def random_string_with_n_digits(n: int) -> str:

    """Generates a random string containing n digits, where n is a positive integer.

    The function ensures that the first digit is never zero.

    The string is returned as a single string rather than a list of characters.



    Args:

        n: The number of digits in the string.



    Returns:

        A random string containing n digits.

    """

    random_string = ''

    while len(random_string) < n:

        digit = str(random.randint(0, 9))

        if len(random_string) == 0 and digit == '0':

            continue  # Skip if the first digit is zero

        random_string += digit



    return random_string

