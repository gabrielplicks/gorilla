import re

from typing import List, Tuple



def split_words_with_length(input_string: str) -> List[Tuple[str, int]]:

    """Splits a string of comma-separated words into a list of tuples, each tuple containing the word and a number of characters.



    Args:

        input_string: The input string to split.



    Returns:

        A list of tuples, each containing a word and its length.

    """

    words = re.findall(r'\w+', input_string)

    tuples = []



    for word in words:

        length = len(word)

        tuples.append((word, length))



    return tuples

