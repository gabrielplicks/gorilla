from itertools import cycle

from typing import Iterable



def generate_padding(n: int) -> bytearray:

    """Creates a byte array of `n` bytes using a specific padding scheme.



    The padding scheme is as follows:

    - The first byte should be `0x00`.

    - The second byte should be `0x01`.

    - The third byte should be `0x02`, and so on until the `n-1` byte, where `n-1` is the last byte.

    - The padding should be repeated until the byte array has a length of `n` bytes.



    Args:

        n: The length of the byte array to generate.



    Returns:

        The final byte array.

    """

    padding = cycle(range(n))

    return bytearray([next(padding) for _ in range(n)])

