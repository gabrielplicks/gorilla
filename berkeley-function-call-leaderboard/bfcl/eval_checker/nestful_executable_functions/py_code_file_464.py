from typing import List



def validate_parens(s: str) -> bool:

    """

    Validates whether the parentheses in a given string are nested correctly.



    Args:

        s: A string containing only parentheses.



    Returns:

        A boolean value indicating whether the parentheses are nested correctly.



    Raises:

        ValueError: If the input string contains invalid characters.

    """

    stack: List[str] = []

    for char in s:

        if char == '(':

            stack.append(char)

        elif char == ')':

            if len(stack) == 0:

                return False

            stack.pop()

        else:

            raise ValueError('Invalid character found')



    return len(stack) == 0

