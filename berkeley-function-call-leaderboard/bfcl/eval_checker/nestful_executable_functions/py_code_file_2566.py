from typing import Set



def duplicate_sentence_checker(sentence: str) -> bool:

    """Checks if a sentence has duplicate words, ignoring punctuation.



    Args:

        sentence: The input sentence.



    Returns:

        True if the sentence has duplicate words, False otherwise.

    """

    punctuation = [".", ",", "'"]

    for p in punctuation:

        sentence = sentence.replace(p, "")

    words = sentence.split()

    unique_words: Set[str] = set()

    for word in words:

        if word in unique_words:

            return True  # Duplicate word detected

        else:

            unique_words.add(word)  # Add the word to the set



    return False  # No duplicate words found

