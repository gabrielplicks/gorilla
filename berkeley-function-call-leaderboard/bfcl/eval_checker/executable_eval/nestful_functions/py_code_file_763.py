from typing import List



def find_lines_with_python(filename: str) -> List[str]:

    """Finds lines in a file that contain the word "Python" and returns them with the word in uppercase.



    Args:

        filename: The name of the file to search.



    Returns:

        A list of lines containing the word "PYTHON".

    """

    with open(filename, 'r') as file:

        lines = []

        for line in file:

            if 'Python' in line:

                line = line.replace('Python', 'PYTHON')

                lines.append(line)

        return lines

