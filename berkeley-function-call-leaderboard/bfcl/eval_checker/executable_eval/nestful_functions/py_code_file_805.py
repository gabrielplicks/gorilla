from typing import List



def check_stopword(word: str, stopwords: List[str]) -> bool:

    """Checks if the given word is a stopword.



    Args:

        word: The word to check.

        stopwords: The list of stopwords.



    Returns:

        True if the word is a stopword, False otherwise.

    """

    return word in stopwords

