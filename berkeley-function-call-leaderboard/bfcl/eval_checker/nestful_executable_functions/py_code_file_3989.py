from typing import Union



def concatenate_strings(s1: Union[str, None], s2: Union[str, None]) -> str:

    """Concatenates two strings. If one of the strings is None or empty, returns the other string.



    Args:

        s1: The first string.

        s2: The second string.

    """

    if s1 is None or not s1:

        return s2

    elif s2 is None or not s2:

        return s1

    else:

        return s1 + s2

