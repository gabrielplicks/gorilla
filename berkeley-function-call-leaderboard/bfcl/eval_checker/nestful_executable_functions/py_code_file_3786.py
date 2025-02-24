from typing import List



def remove_text_between_brackets(s: str) -> str:

    """Removes all text between square brackets (including the brackets) from a string.



    This function handles nested brackets and does not assume that the brackets are always properly closed and nested.



    Args:

        s: The input string.



    Returns:

        The input string with all text between square brackets (including the brackets) removed.

    """

    stack: List[int] = []

    result: List[str] = list(s)

    for i, c in enumerate(s):

        if c == '[':

            stack.append(i)

        elif c == ']':

            start = stack.pop()

            end = i

            result[start] = ''

            result[end] = ''

    return ''.join(result)

