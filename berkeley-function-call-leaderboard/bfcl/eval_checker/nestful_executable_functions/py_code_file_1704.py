from typing import List



def read_lines_containing_python(file_path: str) -> List[str]:

    """Reads the specified file and returns all lines containing "python", "Python", or "PYTHON" (case-insensitive).



    Args:

        file_path: The path to the file to read.



    Returns:

        A list of lines containing "python", "Python", or "PYTHON".

    """

    lines_containing_python = []



    with open(file_path, 'r') as file:

        for line in file:

            if 'python' in line.lower():

                lines_containing_python.append(line.strip())



    return lines_containing_python

