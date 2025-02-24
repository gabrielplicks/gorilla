from typing import Dict



def find_max_word_frequency(word_frequency_map: Dict[str, int]) -> str:

    """Finds the word with the highest frequency in a dictionary.

    If two or more words have the same frequency, returns the word that comes first alphabetically.

    If the frequency is zero, returns an empty string.

    Args:

        word_frequency_map: A dictionary where keys are words and values are their frequencies.

    """

    frequencies = word_frequency_map.values()

    max_frequency = max(frequencies, default=0)



    if max_frequency == 0:

        return ""

    max_word = max(word_frequency_map, key=lambda word: (word_frequency_map[word], word))



    return max_word

