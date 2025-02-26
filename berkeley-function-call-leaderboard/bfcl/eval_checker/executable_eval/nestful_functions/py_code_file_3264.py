from typing import List



def all_suffixes(string: str) -> List[str]:

    """Returns a list of all suffixes of the input string.



    A suffix is a substring that starts from a specific position and extends to the end of the string.



    Args:

        string: The input string.



    Returns:

        A list of all suffixes of the input string.

    """

    suffixes = []

    for i in range(len(string)):

        if string.endswith(string[i:]):

            suffixes.append(string[i:])

    return suffixes

