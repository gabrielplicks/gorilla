from typing import List



def find_first_index(strings: List[str], target: str) -> int:

    """Finds the first index at which a target string appears in a list of strings.



    Args:

        strings: A list of strings.

        target: The target string to find.



    Returns:

        The first index at which the target string appears in the list. If the target string is not found, returns `-1`.

    """

    for i, s in enumerate(strings):

        if s == target:

            return i

    return -1

