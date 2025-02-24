from typing import List, Union



def tokenize_and_capitalize(text: str) -> str:

    """Tokenizes a string into a list of words and capitalizes the first letter of each word.



    Args:

        text: The input string to tokenize and capitalize.



    Returns:

        The input string with the first letter of each word capitalized.

    """

    words: List[str] = text.split()

    capitalized_words: List[str] = [word[0].upper() + word[1:] for word in words]

    return ' '.join(capitalized_words)

