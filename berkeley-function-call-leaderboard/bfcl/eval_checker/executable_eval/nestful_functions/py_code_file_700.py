from typing import Union



def get_degree(n: Union[int, float]) -> int:

    """Calculates the degree of an integer `n`.



    The degree of an integer is the number of bits that are set to 1 in its binary representation.



    Args:

        n: The integer to calculate the degree of.



    Returns:

        The degree of `n`.

    """

    degree = 0



    while n > 0:

        degree += n & 1

        n >>= 1



    return degree

