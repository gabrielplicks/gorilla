from typing import List



def format_content(s: str, words: List[str]) -> str:

    """Combines a string and a list of words into a new string.



    Args:

        s: The original string.

        words: The list of words to be appended to the original string.



    Returns:

        A new string with each word in `words` enclosed in double quotes,

        separated by commas, and appended to the end of the original string `s`.

    """

    word_list = []

    for word in words:

        word_list.append('"' + word + '"')

    combined_string = s + ', ' + ', '.join(word_list)

    return combined_string

