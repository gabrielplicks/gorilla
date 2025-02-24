from typing import List



def list_files_without_subdirectories() -> List[str]:

    """

    Reads a file 'files.txt' and returns a list of strings with no subdirectories.



    Returns:

        A list of strings with no subdirectories.

    """

    file_list = []

    with open('files.txt', 'r') as f:

        for line in f:

            if not '/' in line.strip():

                file_list.append(line.strip())

    return file_list

