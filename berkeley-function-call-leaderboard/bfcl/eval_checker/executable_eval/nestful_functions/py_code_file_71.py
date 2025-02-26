import math



def bitwise_lowest_set_bit_retrieval(n: int) -> int:

    """

    Returns the bit index of the lowest set bit in the binary representation of `n`.

    If `n` is `0`, returns `-1`.



    Args:

        n: A positive integer.



    Returns:

        The bit index of the lowest set bit in the binary representation of `n`.

        If `n` is `0`, returns `-1`.

    """

    if n == 0:

        return -1

    else:

        return int(math.log2(n & -n))

