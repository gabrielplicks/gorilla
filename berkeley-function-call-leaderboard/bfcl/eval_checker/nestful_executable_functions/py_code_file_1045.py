from typing import Union



def str2bin(s: Union[str, bytes]) -> str:

    """Converts a string into the binary representation of each character's ASCII code, using a space to separate each code.



    Args:

        s: The input string.



    Returns:

        A string where each binary code is separated by a space.

    """

    return " ".join([bin(ord(c))[2:] for c in s])

