import os

from typing import List



def cat_files(filenames: List[str]) -> str:

    """Concatenates the contents of a list of files into a single string.

    Args:

        filenames: A list of filenames to read.

    Returns:

        A string containing the contents of all files concatenated together.

    """

    output = ""

    for filename in filenames:

        if not os.path.isfile(filename):

            print(f"File not found: {filename}")

            continue

        if not os.access(filename, os.R_OK):

            print(f"File is not readable: {filename}")

            continue

        with open(filename, "r") as f:

            output += f.read()

    return output

