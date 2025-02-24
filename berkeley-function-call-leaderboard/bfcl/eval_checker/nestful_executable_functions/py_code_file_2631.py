from typing import List



def search_all_words(word_list: List[str], str: str) -> List[str]:

    """Returns a list of words from `word_list` that occur in `str` as a substring.

    Each word in the list is in the same order in which it appears in `str`.

    Note: `str` can contain multiple instances of a word in `word_list`.



    Args:

        word_list: A list of words to search for in `str`.

        str: The string to search for the words in `word_list`.



    Returns:

        A list of words from `word_list` that occur in `str` as a substring.

    """

    output = []

    for word in word_list:

        if word in str:

            output.append(word)

    return output

