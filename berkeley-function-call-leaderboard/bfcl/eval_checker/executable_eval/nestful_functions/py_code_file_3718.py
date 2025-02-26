from typing import Union



def bin_repr(n: Union[int, float]) -> str:

    """Returns the binary representation of the integer `n` as a string with a leading `"0b"` prefix.



    Args:

        n: A positive integer.



    Returns:

        The binary representation of `n` as a string with a leading `"0b"` prefix.

    """

    if n == 0:

        return "0"



    binary_bits = []

    while n > 0:

        binary_bits.append(n % 2)

        n //= 2



    binary_bits.reverse()

    return "0b" + "".join([str(bit) for bit in binary_bits])

