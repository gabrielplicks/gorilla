from typing import Union



def add_quotes(s: str) -> str:

    """Adds single quotes around a string if it doesn't already have them.



    Args:

        s: The input string.



    Returns:

        The input string with single quotes around it if it didn't already have them.

    """

    return s if "'" in s else f"'{s}'"

