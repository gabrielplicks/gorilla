from typing import List



def read_first_three_words(filename: str) -> List[str]:

    """Reads the first three words from a text file and returns them as a list.



    Args:

        filename: The name of the text file.



    Returns:

        A list containing the first three words from the file.

    """

    words = []

    with open(filename, 'r') as file:

        for _ in range(3):

            words.append(next(file))

    return words

