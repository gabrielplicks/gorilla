import os

from typing import List



def change_working_directory() -> str:

    """Changes the current working directory to the current file's directory.

    If the new current working directory is empty, it returns the empty string ("").

    Otherwise, it returns the current working directory.

    """

    current_directory: str = os.getcwd()

    current_file_path: str = os.path.abspath(__file__)

    os.chdir(os.path.dirname(current_file_path))

    if not os.listdir(os.getcwd()):

        return ""

    else:

        return os.getcwd()

