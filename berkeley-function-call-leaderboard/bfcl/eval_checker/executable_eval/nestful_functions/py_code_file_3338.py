from typing import TextIO



def find_line_with_key(file_path: str) -> str:

    """Reads the contents of a text file and returns the first line containing "key".

    If the file does not contain the "key", returns an empty string.

    Args:

        file_path: The path to the text file.

    """

    try:

        with open(file_path, 'r') as file:

            for line in file:

                if 'key' in line:

                    return line.strip()

        return ''

    except FileNotFoundError:

        print(f'File not found: {file_path}')

        return ''

