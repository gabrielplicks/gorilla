from typing import Dict



def calculate_word_frequency_map(text: str) -> Dict[str, int]:

    """Calculates the frequency of each word in a string of text.



    Args:

        text: The input text.



    Returns:

        A dictionary mapping each word to its frequency as an integer.

    """

    text = text.lower()

    text = text.replace(",", "")

    text = text.replace(".", "")

    words = text.split()

    word_frequency_map = {}

    for word in words:

        if word in word_frequency_map:

            word_frequency_map[word] += 1

        else:

            word_frequency_map[word] = 1



    return word_frequency_map

