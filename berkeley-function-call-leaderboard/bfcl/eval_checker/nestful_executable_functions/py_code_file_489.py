def convert_lossy(a: str, b: str) -> str:

    """

    Converts `a` to a string, replacing any non-numeric characters with `b`.

    Args:

        a: The input string to be converted.

        b: The character to replace non-numeric characters in `a`.

    """

    assert isinstance(b, str), "b must be a string"

    result = ""

    for c in a:

        if c.isdigit():

            result += c

        else:

            result += b

    return result

