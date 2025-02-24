from collections import Counter

from typing import Dict



def get_count_dictionary(string: str) -> Dict[str, int]:

    """Creates a dictionary containing the count of each character in the string.



    Args:

        string: The input string.



    Returns:

        A dictionary where the keys are the characters in the string and the values are the counts of each character.

    """

    return Counter(string)

