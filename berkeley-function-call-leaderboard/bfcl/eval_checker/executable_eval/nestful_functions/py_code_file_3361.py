from typing import List



def remove_blank_lines(lines: List[str]) -> List[str]:

    """Removes blank lines from a list of lines.



    Args:

        lines: A list of lines.



    Returns:

        A list of non-blank lines.

    """

    non_blank_lines = []

    for line in lines:

        if line != '\n':

            non_blank_lines.append(line)

    return non_blank_lines

