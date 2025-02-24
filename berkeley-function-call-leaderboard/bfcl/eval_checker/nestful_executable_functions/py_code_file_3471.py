from typing import AnyStr



def remove_newlines_and_replace_with_space(s: AnyStr) -> AnyStr:

    """Removes all newlines from the string and returns the result.

    Replaces the removed newlines with spaces.

    Uses a loop for this operation and does not use the `strip` method.



    Args:

        s: The input string.



    Returns:

        The string with newlines removed and replaced with spaces.

    """

    result = ""

    for char in s:

        if char == "\n":

            result += " "

        else:

            result += char

    return result

