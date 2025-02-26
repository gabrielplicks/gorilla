from typing import AnyStr



def hash_ascii_string(string: AnyStr) -> int:

    """Calculates the hash value of an ASCII string.

    Args:

        string: The ASCII string to be hashed.

    """

    return hash(string)

