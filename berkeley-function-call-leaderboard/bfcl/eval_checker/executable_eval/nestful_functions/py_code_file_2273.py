from typing import List



def isValid(s: str) -> bool:

    """Determines if the parentheses in a string are balanced.



    Args:

        s: The input string consisting of parentheses characters.



    Returns:

        True if the parentheses are balanced, False otherwise.

    """

    stack: List[str] = []

    parentheses = {")": "(", "}": "{", "]": "["}

    for char in s:

        if char in parentheses.values():

            stack.append(char)

        elif char in parentheses:

            if not stack or stack[-1] != parentheses[char]:

                return False

            stack.pop()

        else:

            return False

    return not stack

