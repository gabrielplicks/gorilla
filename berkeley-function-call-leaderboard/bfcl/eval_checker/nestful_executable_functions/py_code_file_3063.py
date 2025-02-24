import re

import typing



def sentence_to_words(sentence: str) -> typing.List[str]:

    """Splits a string into a list of words, removing any punctuation marks.



    Args:

        sentence: The input string to be split into words.



    Returns:

        A list of strings, where each string corresponds to a word in the original string.

    """

    words = sentence.split()  # Split the sentence into a list of words

    words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Remove punctuation marks

    return words

