from typing import List



def formable_words(words: List[str], goal: str) -> List[str]:

    """

    Returns a list of all words in `words` that can be formed using the characters in `goal`.



    Args:

        words: A list of strings.

        goal: A string.



    Returns:

        A list of strings.

    """

    formable_words = []



    for word in words:

        if all(word.count(char) <= goal.count(char) for char in word):

            formable_words.append(word)



    return formable_words

