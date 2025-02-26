from typing import Union



def fibonacci_at_index(index_string: str) -> Union[int, str]:

    """

    Computes the Fibonacci number at the index represented by the given string.



    Args:

        index_string: A string representation of a positive integer.



    Returns:

        The Fibonacci number at the corresponding index.

    """

    index = int(index_string)



    if index <= 1:

        return index



    a, b = 0, 1



    for _ in range(index - 1):

        a, b = b, a + b



    return b

