from typing import Union



def encode_binary(x: int, n: int) -> str:

    """Encodes an integer `x` into a binary string with a minimum of `n` bits.

    Args:

        x: The integer to be encoded.

        n: The desired number of bits in the resulting binary string.

    """

    binary_repr = bin(x)[2:]  # Convert integer to binary string

    padding = n - len(binary_repr)  # Calculate padding length

    result = '0' * padding + binary_repr  # Pad with leading zeros

    return result

