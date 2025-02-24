from typing import List



def extract_words_from_text(text: str) -> List[str]:

    """Extracts all words from a given string text (including non-alphabetical characters).



    The function returns a list of words, where each word is a list of alphanumeric characters.



    Args:

        text: The input string to extract words from.



    Returns:

        A list of words extracted from the input text.

    """

    words = []

    current_word = ''



    for char in text:

        if char.isalnum():

            current_word += char

        elif current_word:

            words.append(current_word)

            current_word = ''



    if current_word:

        words.append(current_word)



    return words

