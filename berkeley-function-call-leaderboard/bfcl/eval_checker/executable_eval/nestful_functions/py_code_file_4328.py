from typing import List



def read_and_parse_file(file_path: str) -> List[str]:

    """Reads and parses a file, returning a list of strings that represent the file content.



    Args:

        file_path: The path to the file to read.



    Returns:

        A list of strings, where each string represents a line in the original file.



    Raises:

        FileNotFoundError: If the file does not exist.

        IOError: If there is an error opening the file.

    """

    try:

        with open(file_path, 'r') as f:

            return f.readlines()

    except FileNotFoundError:

        print("File not found.")

    except IOError:

        print("Error opening file.")

