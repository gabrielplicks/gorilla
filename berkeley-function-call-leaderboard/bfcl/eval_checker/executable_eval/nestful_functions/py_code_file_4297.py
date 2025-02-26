from typing import List



def convert_to_lowercase_except_after_single_quote(input_string: str) -> str:

    """Converts a string to lowercase, except for any character that comes after a single quote.



    Args:

        input_string: The input string to be converted.



    Returns:

        The converted string.

    """

    words: List[str] = input_string.split()

    converted_words: List[str] = []

    for word in words:

        if word[0] == "'":

            converted_words.append(word)

        else:

            converted_words.append(word.lower())

    return ' '.join(converted_words)

