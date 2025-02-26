import string

from typing import List



def read_and_normalize_lines(file_path: str) -> List[str]:

    """Reads a text file and splits it into lines. Normalizes each line by converting it to lowercase and removing all punctuation.



    Args:

        file_path: The path to the file to read.



    Returns:

        A list of all the lowercase and normalized lines.

    """

    with open(file_path, "r") as file:

        text = file.read()



    lines = text.splitlines()

    normalized_lines = [line.lower().translate(str.maketrans('', '', string.punctuation)) for line in lines]



    return normalized_lines

