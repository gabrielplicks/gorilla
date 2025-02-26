from typing import Dict, List



def check_bracket_balance(input_string: str) -> bool:

    """Check if a string of opening and closing brackets is balanced.



    Args:

        input_string: The string to check.



    Returns:

        True if the string is balanced, False otherwise.

    """

    matching_brackets: Dict[str, str] = {

        '}': '{',

        ']': '[',

        ')': '('

    }



    stack: List[str] = []



    for char in input_string:

        if char in matching_brackets.values():

            stack.append(char)

        elif char in matching_brackets:

            if len(stack) == 0:

                return False

            elif stack[-1] == matching_brackets[char]:

                stack.pop()

            else:

                return False



    return len(stack) == 0

