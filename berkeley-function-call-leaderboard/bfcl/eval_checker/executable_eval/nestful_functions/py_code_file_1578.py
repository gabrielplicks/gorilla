from typing import List



def multiply_string(str_input: str, n: int) -> str:

    """Multiplies a string by a given integer.



    Args:

        str_input: The input string to be multiplied.

        n: The number of times to repeat the string.



    Returns:

        The input string repeated `n` times.

    """

    result: List[str] = []

    for _ in range(n):

        result.append(str_input)

    return ''.join(result)

