import sys

import os



def current_file_name_in_dir() -> str:

    """

    Captures the name of the current file and the path to the file's directory and combines them into a single string.



    Returns:

        The name of the current file and the path to the file's directory combined into a single string, in the format `current_file_name_in_dir.txt`.

    """

    current_file_name = os.path.basename(sys.argv[0])

    current_file_path = os.path.abspath(sys.argv[0])

    current_dir_path = os.path.dirname(current_file_path)

    return os.path.join(current_dir_path, current_file_name)

