from collections import defaultdict

from typing import Dict, AnyStr



def count_and_group_characters(string: AnyStr) -> Dict[str, int]:

    """Counts and groups the number of occurrences of each character in a string.



    Args:

        string: The input string.



    Returns:

        A dictionary with keys being unique characters in the string, and values

        being the number of occurrences.

    """

    string = str(string)

    character_counts = defaultdict(int)

    for character in string:

        character_counts[character] += 1

    return dict(character_counts)

