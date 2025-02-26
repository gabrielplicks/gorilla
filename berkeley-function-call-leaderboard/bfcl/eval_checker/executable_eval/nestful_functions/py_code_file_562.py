from typing import TextIO



def read_file_uppercase(filename: str) -> str:

    """Reads a file line by line and returns the text in uppercase.



    Args:

        filename: The name of the file to read.

    """

    result = []

    with open(filename, 'r') as f:

        for line in f:

            result.append(line.upper())

    return "".join(result)

