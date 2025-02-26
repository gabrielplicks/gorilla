from typing import Union



def zeroes_to_binary(n: Union[int, float]) -> str:

    """

    Converts a positive integer `n` to a binary string of `n` zeros.



    Args:

        n: A positive integer.



    Returns:

        A binary string of `n` zeros.

    """

    return format(0, '0{}b'.format(n))

