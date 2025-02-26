import re

from typing import List, Tuple



def get_substrings_with_positions(string: str) -> List[Tuple[str, int]]:

    """

    Returns a list of tuples containing substrings that are all letters and their positions in the given string.



    Args:

        string: The input string.



    Returns:

        A list of tuples containing substrings that are all letters and their positions in the given string.

    """

    substrings = re.findall(r'[a-zA-Z]+', string)

    positions = [match.start() for match in re.finditer(r'[a-zA-Z]+', string)]

    return list(zip(substrings, positions))

