import os

from typing import List



def directory_traversal(directory_path: str) -> int:

    """Traverses a given directory and its subdirectories, and returns the number of files whose name starts with the word `test` or `Test`.



    Args:

        directory_path: The path of the directory to traverse.



    Returns:

        The number of files whose name starts with the word `test` or `Test`.

    """

    for current_directory, subdirectories, files in os.walk(directory_path):

        test_files = [file for file in files if file.startswith('test') or file.startswith('Test')]

        return len(test_files)

