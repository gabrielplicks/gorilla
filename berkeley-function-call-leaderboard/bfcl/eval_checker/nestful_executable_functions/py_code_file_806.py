from typing import List



def read_lines_reversed(file_path: str) -> List[str]:

    """Read a text file line by line, outputting each line in reverse order.

    If the file is not found, return an empty list.

    Args:

        file_path: The path to the file.

    Returns:

        A list of reversed lines.

    """

    lines_reversed = []



    try:

        with open(file_path) as file:

            for line in file:

                lines_reversed.append(line[::-1])

    except FileNotFoundError:

        print("File not found.")



    return lines_reversed

