from typing import Dict



def word_frequency_from_string(string: str) -> Dict[str, int]:

    """Calculates the frequency of words in a string.



    Args:

        string: The input string.



    Returns:

        A dictionary of the word frequency in the string.

    """

    string = ''.join(char for char in string if char.isalnum() or char.isspace())

    string = string.lower()

    words = string.split()

    freq = {}

    for word in words:

        freq[word] = freq.get(word, 0) + 1

    return freq

