from typing import List



def read_file_ignore_header_footer(filename: str) -> List[str]:

    """Reads a file line by line, ignoring the first two lines (the header and the footer), and returns a list of all the lines.



    Args:

        filename: The name of the file to read.



    Returns:

        A list of all the lines in the file, excluding the first two lines.

    """

    lines = []

    with open(filename, "r") as file:

        next(file)

        next(file)

        for line in file:

            lines.append(line)



    return lines

