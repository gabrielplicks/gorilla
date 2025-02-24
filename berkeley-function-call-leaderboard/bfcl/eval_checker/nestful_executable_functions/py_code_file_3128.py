from typing import AnyStr



def check_parens(string: AnyStr) -> bool:

    """Checks if the parentheses in a string are balanced or not.



    Args:

        string: The input string to check.



    Returns:

        True if the parentheses are balanced, False otherwise.

    """

    open_parens = 0

    for char in string:

        if char == '(':

            open_parens += 1

        elif char == ')':

            if open_parens == 0:

                return False

            open_parens -= 1

    return open_parens == 0

