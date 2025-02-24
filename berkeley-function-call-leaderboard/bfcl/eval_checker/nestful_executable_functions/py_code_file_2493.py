from typing import Union



def convert_reversed(n: Union[int, float]) -> str:

    """Converts a positive integer `n` to its decimal representation as a string, starting from the least significant digit.



    Args:

        n: A positive integer.



    Returns:

        The decimal representation of `n` as a string, starting from the least significant digit.

    """

    output = ''

    while n > 0:

        digit = n % 10

        output += str(digit)

        n //= 10

    return output

