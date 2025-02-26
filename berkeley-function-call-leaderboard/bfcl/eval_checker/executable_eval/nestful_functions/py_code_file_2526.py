import os

from typing import List



def sort_by_content(files: List[str]) -> List[str]:

    """Sorts a list of file paths based on the content of each file.



    Args:

        files: A list of file paths.



    Returns:

        A sorted list of file paths.

    """

    def sort_key(file_path: str) -> tuple:

        """Sorting key function that reads the content of a file and returns a tuple containing the content and the file path.



        Args:

            file_path: A file path.



        Returns:

            A tuple containing the content of the file and the file path.

        """

        with open(file_path, 'r') as f:

            content = f.read()

        return (content, file_path)



    sorted_files = sorted(files, key=sort_key)

    return sorted_files

