from typing import List, Dict



def count_words_from_sentences(sentences: List[str]) -> Dict[str, int]:

    """Counts the words in a list of sentences.



    Args:

        sentences: A list of sentences.



    Returns:

        A dictionary of words and their corresponding counts.

    """

    word_counts = {word: 0 for sentence in sentences for word in sentence.split()}

    for sentence in sentences:

        for word in sentence.split():

            word_counts[word] += 1

    return word_counts

