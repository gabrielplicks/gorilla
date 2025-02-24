from typing import List



def count_a_words(sentence: str) -> int:

    """Counts the number of words in a sentence that begin with the letter "a".



    Args:

        sentence: The sentence to count words in.



    Raises:

        TypeError: If the input is not a string.

    """

    if not isinstance(sentence, str):

        raise TypeError('Input must be a string')



    words: List[str] = sentence.split()

    a_words: List[str] = [word for word in words if word.startswith('a')]

    return len(a_words)

