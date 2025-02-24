from typing import AnyStr



def hash_string_cast(s: AnyStr) -> float:

    """Calculates the hash value of casting the string to a float.



    Args:

        s: The input string.



    Returns:

        The hash value of the input string as a float.

    """

    return float(hash(s))

