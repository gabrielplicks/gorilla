from typing import List



def extract_comments(file_path: str) -> List[str]:

    """Extracts comment lines from a Python source code file.



    Args:

        file_path: The path to the Python source code file.



    Returns:

        A list of comment lines.

    """

    with open(file_path, 'r') as f:

        lines = f.readlines()

    comment_lines = []

    for line in lines:

        if line.startswith('#') or line.startswith('//'):

            comment_lines.append(line.strip())

    return comment_lines

