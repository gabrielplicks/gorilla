from typing import List



def get_duplicates(words: List[str]) -> List[str]:

    """Returns a new list containing all words that are duplicates.

    Args:

        words: A list of words.

    """

    duplicates = []

    for i in range(1, len(words)):

        for j in range(i):

            if words[i] == words[j]:

                duplicates.append(words[i])

                break

    return duplicates

