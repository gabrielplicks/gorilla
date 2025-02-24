from typing import List



def find_common_lines(file_path_1: str, file_path_2: str) -> List[str]:

    """Finds and returns the common lines in two files.



    Args:

        file_path_1: The path to the first file.

        file_path_2: The path to the second file.

    """

    common_lines = []

    lines_1 = set()



    with open(file_path_1, 'r') as file_1:

        for line in file_1:

            lines_1.add(line)



    with open(file_path_2, 'r') as file_2:

        for line in file_2:

            if line in lines_1:

                common_lines.append(line)



    return common_lines

