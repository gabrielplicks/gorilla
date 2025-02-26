from typing import List



def is_balanced(parens: str) -> bool:

    """Checks if the parentheses in a string are balanced.



    Args:

        parens: A string containing parentheses.



    Returns:

        True if the parentheses are balanced, False otherwise.

    """

    stack: List[str] = []

    for char in parens:

        if char in ['(', '[', '{']:

            stack.append(char)

        else:

            if not stack:

                return False

            if char == ')' and stack[-1] != '(':

                return False

            if char == ']' and stack[-1] != '[':

                return False

            if char == '}' and stack[-1] != '{':

                return False

            stack.pop()

    return len(stack) == 0

