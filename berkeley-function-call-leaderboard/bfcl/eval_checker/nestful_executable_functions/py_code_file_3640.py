from typing import Tuple



def retain_char(s: str, n: int) -> str:

    """Retains only the characters in a string whose position is divisible by an integer.



    Args:

        s: The input string.

        n: The integer divisor.



    Returns:

        The resulting string.

    """

    result = ""

    counter = 1



    for char in s:

        if counter % n == 0:

            result += char

        counter += 1



    return result

