from typing import List



def join_bytes_with_separator(input: List[bytes], separator: bytes) -> bytes:

    """Joins a list of bytes-like objects with a separator string.



    Args:

        input: A list of bytes-like objects.

        separator: A separator string.



    Returns:

        A single bytes-like object containing the input bytes-like objects

        concatenated with the separator string.

    """

    output = b''

    for i in range(len(input)):

        output += input[i]

        if i < len(input) - 1:

            output += separator

    return output

