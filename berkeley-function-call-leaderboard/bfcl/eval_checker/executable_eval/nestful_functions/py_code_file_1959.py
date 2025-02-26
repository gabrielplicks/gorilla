def repeat_string_python2(string: str, n: int) -> str:

    """Repeats a string n times in Python 2.

    Args:

        string: The string to repeat.

        n: The number of times to repeat the string.

    """

    return string * n

def repeat_string_python3(string: str, n: int) -> str:

    """Repeats a string n times in Python 3.

    Args:

        string: The string to repeat.

        n: The number of times to repeat the string.

    """

    if not isinstance(n, int):

        raise TypeError('n must be an integer')

    return string * n

