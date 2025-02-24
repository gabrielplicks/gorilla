from collections import OrderedDict

from typing import List



def merge_and_remove_duplicates(strings: List[str]) -> List[str]:

    """Merges a list of strings and removes duplicate characters from each string.



    The order of strings and characters within each string is preserved.



    Args:

        strings: A list of strings containing only lowercase alphanumeric characters.



    Returns:

        A list of strings with duplicate characters removed.

    """

    merged_string = ''.join(strings)

    unique_characters = OrderedDict.fromkeys(merged_string)

    return [''.join(unique_characters)]

