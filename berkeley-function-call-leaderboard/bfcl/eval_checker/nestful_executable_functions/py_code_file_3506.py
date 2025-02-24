from collections import Counter

from typing import Dict



def get_term_freq_dict(text: str) -> Dict[str, int]:

    """Calculates the term frequencies in a given text.



    Args:

        text: The input text.



    Returns:

        A dictionary with the words as keys and their respective frequencies as values.

    """

    words = text.split()

    term_freq_dict = Counter(words)

    return term_freq_dict

