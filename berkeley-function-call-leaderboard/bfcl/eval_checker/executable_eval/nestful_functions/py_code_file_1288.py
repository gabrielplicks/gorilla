from typing import List



def convert_bits(n: int) -> List[int]:

    """Converts an integer to a list of bits using a bit mask and a left shift operation.



    Args:

        n: A non-negative integer.



    Returns:

        A list of bits.

    """

    mask = 1

    bit_list = []



    while mask <= n:

        bit_list.append(mask & n)

        mask <<= 1



    return bit_list

