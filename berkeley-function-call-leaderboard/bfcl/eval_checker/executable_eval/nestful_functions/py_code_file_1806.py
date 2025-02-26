from typing import Dict



def create_dictionary_from_file(filename: str) -> Dict[str, str]:

    """

    Creates a dictionary from a text file. The text file is a list of key-value pairs,

    with each line containing a pair and separated by a colon (:).

    Args:

        filename: The name of the file to read from.

    """

    dictionary = {}



    with open(filename) as f:

        for line in f:

            key, value = line.strip().split(':')

            dictionary[key] = value



    return dictionary

