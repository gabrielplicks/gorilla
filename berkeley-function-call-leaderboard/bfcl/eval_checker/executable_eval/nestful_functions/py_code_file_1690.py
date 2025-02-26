from typing import List



def join_words(words: List[str]) -> str:

    """Joins a list of word strings into a single string with all words joined by spaces, where words that contain spaces are enclosed in double quotes.



    Args:

        words: A list of word strings.



    Returns:

        A single string with all words joined by spaces, where words that contain spaces are enclosed in double quotes.

    """

    return ' '.join(f'"{word}"' if ' ' in word else word for word in words)

