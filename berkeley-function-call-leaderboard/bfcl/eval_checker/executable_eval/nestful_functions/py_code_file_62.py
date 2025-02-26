from typing import Dict



def word_frequency_dict(sentence: str) -> Dict[str, int]:

    """Calculates the frequency of words in a sentence and returns a sorted dictionary.

    Args:

        sentence: The input sentence.

    Returns:

        A sorted dictionary of words and their frequencies.

    """

    words = sentence.split()

    word_freq = {"UNK": 0, "PAD": 0}

    for word in words:

        word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)



    return dict(sorted_words)

