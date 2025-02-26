from typing import Dict, List



def create_inverted_index(documents: List[List[str]]) -> Dict[str, List[int]]:

    """Creates an inverted index from a list of documents.



    An inverted index is a dictionary mapping terms to documents.



    Args:

        documents: A list of documents, where each document is a list of words.



    Returns:

        A dictionary mapping terms to documents.

    """

    inverted_index: Dict[str, List[int]] = {}

    for i, document in enumerate(documents):

        for word in document:

            if word not in inverted_index:

                inverted_index[word] = []

            inverted_index[word].append(i)

    return inverted_index

