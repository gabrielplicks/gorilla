from typing import List



def read_lines_containing(file_path: str, s: str) -> List[str]:

    """Reads a file and returns all lines that contain the given string `s`.



    Args:

        file_path: The path to the file.

        s: The string to search for in each line.

    """

    result = []



    with open(file_path, 'r') as f:

        for line in f:

            if s in line:

                result.append(line)



    return result

