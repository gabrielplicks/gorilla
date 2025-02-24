from typing import Union



def get_first_line(file_path: str) -> Union[str, None]:

    """

    Reads the first non-empty line of a file.



    Args:

        file_path: The path to the file.



    Returns:

        The first non-empty line of the file, or None if the file is empty or does not exist.

    """

    try:

        with open(file_path) as file:

            first_line = file.readline()

            if not first_line:

                raise Exception("File is empty")

            return first_line

    except FileNotFoundError:

        print("File not found!")

        return None

    except Exception as e:

        print(f"Error: {e}")

        return None

