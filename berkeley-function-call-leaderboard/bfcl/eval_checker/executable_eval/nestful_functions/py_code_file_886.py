from typing import AnyStr



def read_file_by_line_and_get_lines_ending_with_newline(filename: str) -> AnyStr:

    """Reads a file line-by-line and returns a string where each line is terminated by a newline character.



    Args:

        filename: The file path.



    Returns:

        A string where each line is terminated by a newline character.

    """

    with open(filename, 'r') as f:

        lines = []

        while True:

            line = f.readline()

            if line:

                lines.append(line)

            else:

                break



    return ''.join(lines) + '\n'

