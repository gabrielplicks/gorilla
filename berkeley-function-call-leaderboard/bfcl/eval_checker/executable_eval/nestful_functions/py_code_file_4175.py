from typing import List



def merge_lines(file1_path: str, file2_path: str) -> List[str]:

    """Merges the lines of two files into a single list of lines.



    Args:

        file1_path: The path to the first file.

        file2_path: The path to the second file.



    Returns:

        A list of all lines contained in the first file followed by all lines contained in the second file.

    """

    with open(file1_path, 'r') as file1:

        lines = file1.readlines()



    with open(file2_path, 'r') as file2:

        lines.extend(file2.readlines())



    return lines

