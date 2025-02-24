from typing import Union



def bit_to_index(n: Union[int, float]) -> int:

    """

    Returns the zero-based index of the rightmost set bit in binary representation of `n`.

    If `n` is zero, then the function returns `-1`.



    Args:

        n: A 32-bit integer value.

    """

    if n == 0:

        return -1



    mask = 1

    index = 0

    while n & mask == 0:

        n >>= 1

        index += 1



    return index

