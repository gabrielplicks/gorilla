from typing import List



def validate_curly_braces(string: str) -> bool:

    """Validates if an input string has matching curly braces.



    Args:

        string: The input string to validate.



    Returns:

        True if the string has matching curly braces, False otherwise.



    Raises:

        Exception: If the string has unmatched curly braces.

    """

    stack: List[str] = []



    for character in string:

        if character == '{':

            stack.append(character)

        elif character == '}':

            if len(stack) == 0:

                return False

            elif stack[-1] != '{':

                raise Exception("Unmatched curly braces")

            else:

                stack.pop()

    if len(stack) == 0:

        return True

    else:

        return False

