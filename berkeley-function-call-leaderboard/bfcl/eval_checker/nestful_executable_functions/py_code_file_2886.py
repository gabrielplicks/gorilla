import struct

from typing import List



def binary_packed(binary_strings: List[str]) -> bytes:

    """Produces an array of 8-bit integers (bytestrings) that are the binary-packed versions of the input numbers.



    Args:

        binary_strings: A list of binary numbers represented in strings.



    Returns:

        A bytes object containing the binary-packed versions of the input numbers.

    """

    integers = [int(binary_string, 2) for binary_string in binary_strings]

    return struct.pack('B' * len(integers), *integers)

