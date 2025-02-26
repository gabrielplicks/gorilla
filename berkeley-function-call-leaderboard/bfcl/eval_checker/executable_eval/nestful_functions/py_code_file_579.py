from typing import List



def count_common_words(word_list: List[str]) -> int:

    """Counts the number of words that occur more than once in a given list of words.



    Args:

        word_list: A list of words.



    Returns:

        The count of words that occur more than once.

    """

    count = 0

    seen = set()

    for word in word_list:

        if word.lower() in seen:

            count += 1

        else:

            seen.add(word.lower())

    return count

