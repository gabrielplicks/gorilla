from typing import List, Union



def clean_word(text: str) -> str:

    """Removes special characters from a given string, preserving the original case of the text.



    Args:

        text: The input string.



    Returns:

        The cleaned string.

    """

    special_characters: List[str] = ['\'', '"', ',', ';', '?', '!', ':', '.', '(', ')', '[', ']', '{', '}',

                                     '*', '@', '#', '$', '%', '^', '&', '/', '\\']

    clean_text: str = ''

    for character in text:

        if character not in special_characters:

            clean_text += character

    return clean_text

