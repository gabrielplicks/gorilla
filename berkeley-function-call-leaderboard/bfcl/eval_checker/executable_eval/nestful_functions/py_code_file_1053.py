from typing import AnyStr



def create_output(n: int) -> AnyStr:

    """Generates a multiline string of length `n` consisting of multiple lines, where each line contains a single character repeated `n` times.



    Args:

        n: The length of the string and the number of times the character is repeated in each line.



    Returns:

        The generated multiline string.

    """

    output = ''

    for _ in range(n):

        output += '*' * n + '\n'

    return output

