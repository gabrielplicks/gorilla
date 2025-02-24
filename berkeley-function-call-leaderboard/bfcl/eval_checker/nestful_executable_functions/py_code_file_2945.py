from typing import Union



def quote(string: str) -> str:

    """Quotes a string by enclosing it in angle brackets.



    Args:

        string: The string to be quoted.



    Returns:

        The quoted string.

    """

    quoted = f'<{string}>'

    return quoted



def unquote(quoted_string: str) -> str:

    """Unquotes a string by removing the angle brackets.



    Args:

        quoted_string: The quoted string to be unquoted.



    Returns:

        The unquoted string.

    """

    unquoted = quoted_string[1:-1]

    return unquoted

