from typing import Union



def bit_length(n: Union[int, float]) -> int:

    """Calculates the number of bits required to represent an integer in binary.

    Args:

        n: The integer to calculate the bit length for.

    """

    if n == 0:

        return 0

    if n < 0:

        n = -(n + 1)  # find two's complement

        negative = True

    else:

        negative = False

    bits = 0

    while n > 0:

        bits += 1

        n >>= 1  # shift right by 1 bit

    if negative:

        bits += 1  # add 1 bit for the sign

    return bits

