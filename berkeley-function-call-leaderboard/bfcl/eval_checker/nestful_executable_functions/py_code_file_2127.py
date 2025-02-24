from typing import Union



def reversed_string_with_message(s: str) -> str:

    """Reverses a string and appends a message indicating its length.



    Args:

        s: The input string.



    Returns:

        The reversed string with a message indicating its length.

    """

    reversed_s = s[::-1]

    length = len(s)

    message = f"is long" if length > 5 else "is short"

    return f"{reversed_s} {message}"

