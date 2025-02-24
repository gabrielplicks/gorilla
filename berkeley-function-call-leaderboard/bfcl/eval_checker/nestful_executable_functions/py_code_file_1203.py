from typing import List



def get_simple_words_from_string(string: str) -> List[str]:

    """Returns a list of simple words from the string, including the first and last word.

    Simple words are words that do not contain any non-alphabetic characters.

    Args:

        string: The input string.

    Raises:

        Exception: If the input string does not contain at least one word.

    """

    if not string.isalpha():

        raise Exception("Input string must contain at least one word.")

    cleaned_string = "".join(c for c in string if c.isalpha())

    words = cleaned_string.split()

    return [word for word in [words[0], words[-1]] if word.isalpha()]

