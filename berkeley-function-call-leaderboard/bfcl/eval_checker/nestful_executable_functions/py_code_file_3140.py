from typing import Dict, List



def validate_brackets(sequence: str) -> bool:

    """Determines whether a sequence of brackets is valid.



    Args:

        sequence: A string containing brackets.



    Returns:

        True if the sequence is valid, False otherwise.

    """

    stack: List[str] = []

    brackets: Dict[str, str] = {')': '(', '}': '{', ']': '['}



    for line in sequence.splitlines():

        for char in line:

            if char in brackets.values():

                stack.append(char)

            elif char in brackets:

                if not stack or stack.pop() != brackets[char]:

                    return False



    return not stack

