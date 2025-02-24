from typing import List



def first_words(strings: List[str]) -> List[str]:

    """Returns a list of strings that contain only the first word of each string.



    Args:

        strings: A list of strings.



    Returns:

        A list of strings that contain only the first word of each string.

    """

    first_words_list = []

    for string in strings:

        words = string.split()

        if words:

            first_words_list.append(words[0])

    return first_words_list

