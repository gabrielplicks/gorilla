from typing import List



def read_lines_as_words(file_name: str) -> List[str]:

    """Reads a file and returns a list of words.



    Args:

        file_name: The name of the file to read.



    Returns:

        A list of words that appear in the file.

    """

    words = []

    with open(file_name, "r") as file_object:

        for line in file_object:

            line = line.rstrip()

            line_words = line.split()

            words.extend(line_words)

    return words

