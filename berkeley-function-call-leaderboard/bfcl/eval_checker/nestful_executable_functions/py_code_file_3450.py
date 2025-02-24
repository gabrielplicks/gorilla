from typing import List



def load_words(filename: str) -> List[str]:

    """Loads a plain text file into a list of unique, lowercase words.



    Args:

        filename: The name of the file to load.



    Returns:

        A list of unique, lowercase words.

    """

    with open(filename, 'r') as f:

        word_list = []

        for line in f:

            for word in line.split():

                word_list.append(word.lower())

    return list(set(word_list))

