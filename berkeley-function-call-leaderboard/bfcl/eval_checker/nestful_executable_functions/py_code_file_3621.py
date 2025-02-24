from typing import Union



def max_zeros(n: Union[int, float]) -> int:

    """Finds the maximal number of consecutive zeros in the binary representation of `n`.



    Args:

        n: A non-negative integer.



    Returns:

        The maximal number of consecutive zeros in the binary representation of `n`.

    """

    max_count = 0

    count = 0



    while n:

        if n & 1 == 0:  # Check if the least significant bit is 0

            count += 1

        else:  # If the least significant bit is 1, reset the count

            count = 0



        max_count = max(max_count, count)

        n >>= 1  # Shift right by 1 bit



    return max_count

