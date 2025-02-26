from typing import List



def valid_parentheses(s: str) -> bool:

    """Checks whether a string `s` only contains valid parentheses.



    Args:

        s: The string to check.



    Returns:

        True if `s` is valid, and False otherwise.

    """

    stack: List[str] = []

    for char in s:

        if char in "([":

            stack.append(char)

        elif char == ")":

            if not stack or stack.pop() != "(":

                return False

        elif char == "]":

            if not stack or stack.pop() != "[":

                return False

    return len(stack) == 0

