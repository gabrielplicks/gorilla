import re

from typing import List



def extract_unique_words(sentence: str) -> str:

    """Extracts unique words from a sentence and outputs them in a human-readable format.



    If an empty string is passed to the function, it returns a string containing only the message "No words found.".



    Args:

        sentence: The input sentence.



    Returns:

        A string containing the unique words from the input sentence, separated by commas and spaces.

    """

    if not sentence:

        return 'No words found.'



    words = sorted(set(re.findall(r'\w+', sentence)))

    return ', '.join(words)

