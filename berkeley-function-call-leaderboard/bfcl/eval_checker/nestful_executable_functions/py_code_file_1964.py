from typing import AnyStr



def get_str_hash(str: AnyStr) -> int:

    """Computes the hash value of a string and handles hash collisions by adding the length of the string.



    Args:

        str: The input string.



    Returns:

        The hash value of the string plus its length.

    """

    return hash(str) + len(str)

