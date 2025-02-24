from typing import List, Union



def apply_camel_case(text: str) -> str:

    """Applies camel case formatting to a string of words.



    Args:

        text: A string of words separated by spaces.



    Returns:

        The formatted string with camel case applied.

    """

    words: List[str] = text.split()  # Split the string into individual words

    capitalized_words: List[str] = [word[0].upper() + word[1:].lower() for word in words]  # Capitalize the first letter of each word, convert all other letters to lowercase

    return ''.join(capitalized_words)  # Join the words together and return the formatted string

