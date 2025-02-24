from typing import Union



def str_concat(str1: str, str2: str) -> str:

    """Concatenates two strings.

    Args:

        str1: The first string.

        str2: The second string.

    """

    return str1 + str2



def str_format(str1: str, str2: str) -> str:

    """Concatenates two strings with a space in between.

    Args:

        str1: The first string.

        str2: The second string.

    """

    return " ".join([str1, str2])

