from typing import List



def read_file_process_lines(file_name: str) -> List[str]:

    """Reads the contents of a text file, processes each line, and returns the result as a list of lines.

    Handles any exceptions that may occur during the operation and ensures that the file is properly closed after reading.



    Args:

        file_name: The name of the file to read.



    Returns:

        A list of processed lines.

    """

    lines = []



    try:

        with open(file_name, 'r') as file:

            for line in file:

                lines.append(line)



        return lines

    except Exception as e:

        print(f'Error occurred while reading file: {e}')

        return []

