from typing import List



def read_lines_with_pattern(file_path: str, pattern: str) -> List[str]:

    """Reads a text file line-by-line and returns a list of all the lines that contain a given pattern.



    Args:

        file_path: The path to the file.

        pattern: The pattern to search for in each line.

    """

    lines = []



    with open(file_path, 'r') as file:

        for line in file:

            if pattern in line:

                lines.append(line)



    return lines

