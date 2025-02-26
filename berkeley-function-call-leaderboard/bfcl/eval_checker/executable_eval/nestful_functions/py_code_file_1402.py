from typing import Dict



def build_char_dict(word: str) -> Dict[str, int]:

    """Builds a dictionary where the keys are the unique characters in the string and the values are their corresponding minimum indices in the string.



    Args:

        word: The input string.



    Returns:

        A dictionary where the keys are the unique characters in the string and the values are their corresponding minimum indices in the string.

    """

    d = {}

    for i, char in enumerate(word):

        if char in d:

            d[char] = min(d[char], i)

        else:

            d[char] = i

    return d

