from typing import List, Tuple



def find_word_pairs(file_path: str) -> List[Tuple[str, str]]:

    """Reads a file and returns a list of tuples containing the first and last word of each line.



    Args:

        file_path: The path to the file to read.



    Returns:

        A list of tuples, where each tuple contains the first and last word of a line in the file.

    """

    with open(file_path, 'r') as file:

        lines = file.readlines()



    word_pairs = []



    for line in lines:

        words = line.split()

        first_word = words[0]

        last_word = words[-1]

        word_pairs.append((first_word, last_word))



    return word_pairs

