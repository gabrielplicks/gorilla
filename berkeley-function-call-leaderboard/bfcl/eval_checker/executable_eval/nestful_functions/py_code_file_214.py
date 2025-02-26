from typing import Dict



def word_count_and_length(text: str) -> Dict[str, Dict[str, int]]:

    """Calculates the word counts and lengths in a given string.



    Args:

        text: The input string.



    Returns:

        A dictionary of word counts and lengths.

    """

    words = text.split()

    word_counts = {}



    for word in words:

        word = word.lower()

        length = len(word)

        if word in word_counts:

            word_counts[word]['count'] += 1

        else:

            word_counts[word] = {'count': 1, 'length': length}



    return word_counts

