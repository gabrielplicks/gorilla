from typing import Union



def reversed_equal(n: Union[int, float]) -> bool:

    """Checks if the string representations of `n` and its reversed version are equal.



    Args:

        n: A non-negative integer or float.



    Returns:

        True if the string representations of `n` and its reversed version are equal, False otherwise.

    """

    n_string = str(n)

    return n_string == n_string[::-1]

