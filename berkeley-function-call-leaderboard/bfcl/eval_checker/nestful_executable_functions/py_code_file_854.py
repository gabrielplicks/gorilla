from typing import List



def get_file_words(text: str) -> List[str]:

    """

    Returns a list of all the words in the file represented by the given file path.

    If the file does not exist, raises an exception with the message 'File does not exist.'



    Args:

        text: The file path.

    """

    with open(text, 'r') as f:

        words = f.read().split()

    return words

