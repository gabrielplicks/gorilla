from typing import Tuple



def index(tup: Tuple, s: str) -> int:

    """Traverses through the tuple and returns the index of the first element that matches the given string.

    If the string is not found, returns -1.

    Args:

        tup: The tuple to be traversed.

        s: The string to be matched.

    """

    for i, item in enumerate(tup):

        if item == s:

            return i

    return -1

