from typing import Dict



def load_vocabulary_from_text_file(file_path: str) -> Dict[str, int]:

    """Loads a vocabulary from a text file and returns a dictionary mapping each word to an integer ID.



    The input text file should contain one word per line, and the integer IDs are assigned in ascending order.

    The function strips any leading or trailing whitespace from the words and converts them to lowercase before

    mapping to their IDs.



    Args:

        file_path: The path to the text file containing the vocabulary.



    Returns:

        A dictionary mapping each word in the vocabulary to an integer ID.

    """

    with open(file_path) as f:

        file_contents = f.read()

    words = file_contents.split()

    vocabulary = {}

    for i, word in enumerate(words):

        word = word.strip().lower()

        vocabulary[word] = i



    return vocabulary

