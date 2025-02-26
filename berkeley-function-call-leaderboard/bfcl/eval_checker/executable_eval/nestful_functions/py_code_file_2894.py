from typing import List



def concatenate_with_length_prefix(strings: List[bytes]) -> bytes:

    """Concatenates a list of byte strings together, with each concatenated string having a 32-bit length prefix at its beginning.



    Args:

        strings: A list of byte strings to be concatenated.



    Returns:

        The concatenated byte string.

    """

    concatenated_string = b""

    for string in strings:

        length_prefix = len(string).to_bytes(4, "big")

        concatenated_string += length_prefix + string

    return concatenated_string

