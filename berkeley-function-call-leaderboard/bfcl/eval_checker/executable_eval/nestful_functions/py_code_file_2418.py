from collections import Counter

from typing import Dict



def char_freq_table(text: str) -> Dict[str, int]:

    """Creates a frequency table of characters in the input text.



    Args:

        text: The input text.



    Returns:

        A dictionary with each unique character in `text` as a key and a count of the number of occurrences in the string as the value.

    """

    return Counter(text)

