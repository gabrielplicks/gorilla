from typing import Union



def count_1_bits(n: Union[int, float]) -> int:

    """Counts the number of 1 bits in a number `n`.



    Args:

        n: The number to count the 1 bits in.



    Returns:

        The number of 1 bits in `n`.

    """

    num_bits = 0

    while n:

        n &= n - 1

        num_bits += 1

    return num_bits

