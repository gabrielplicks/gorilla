from typing import List, Dict



def word_counts(sentences: List[List[str]]) -> Dict[str, int]:

    """Calculates the counts of words in a list of sentences.



    Args:

        sentences: A list of lists where each sublist represents a sentence and its words.



    Returns:

        A dictionary where the keys are words and values are the counts in the sentences.

    """

    flattened_list = [word for sublist in sentences for word in sublist]

    word_counts = {}

    for word in set(flattened_list):

        word_counts[word] = flattened_list.count(word)

    return word_counts

