from typing import List



def hex_format(n: int) -> str:

    """Converts a positive integer `n` into its hexadecimal representation as a string.



    Args:

        n: The positive integer to convert.



    Returns:

        The hexadecimal representation of `n` as a string.

    """

    digits = '0123456789abcdef'

    result: List[str] = []

    while n > 0:

        remainder = n % 16

        result.append(digits[remainder])

        n //= 16

    return "".join(reversed(result))

