from typing import Union



def unicode_concat(a: str, b: str, c: str) -> Union[str, None]:

    """Concatenates 3 unicode strings a, b and c in the order a, b, c.

    If a is longer than b + c, raises an exception.

    If a, b, and c are all empty strings, returns an empty string.



    Args:

        a: The first unicode string.

        b: The second unicode string.

        c: The third unicode string.



    Raises:

        Exception: If a is longer than b + c.

    """

    if len(a) > len(b) + len(c):

        raise Exception("a is longer than b + c")

    elif not a and not b and not c:

        return ""

    else:

        return a + b + c

