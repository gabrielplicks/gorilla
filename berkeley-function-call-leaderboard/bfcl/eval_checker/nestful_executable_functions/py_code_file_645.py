from typing import TextIO



def display_file_contents(file_path: str) -> int:

    """Displays the contents of a file and returns the number of lines in the file.



    Args:

        file_path: The path to the file to be displayed.



    Returns:

        The number of lines in the file.

    """

    line_count = 0

    with open(file_path, 'r') as file:

        for line in file:

            print(line.strip())

            line_count += 1

    return line_count

