from typing import Tuple



def count_leading_and_trailing_spaces(s: str) -> Tuple[int, int]:

    """Calculates the number of leading and trailing spaces in a string.



    Args:

        s: The input string.



    Returns:

        A tuple containing the number of leading spaces and the number of trailing spaces.

    """

    if not s:

        return (0, 0)

    stripped_leading = s.lstrip()

    stripped_trailing = s.rstrip()

    count_leading = len(s) - len(stripped_leading)

    count_trailing = len(s) - len(stripped_trailing)

    return (count_leading, count_trailing)

