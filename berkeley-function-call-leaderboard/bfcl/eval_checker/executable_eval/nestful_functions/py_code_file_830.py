from typing import List



def add_prefix_to_filenames(file_names: List[str], prefix: str) -> List[str]:

    """Adds a prefix to each file name in a list of file names.



    Args:

        file_names: A list of file names.

        prefix: A string to add to the beginning of each file name.



    Returns:

        A new list of file names with the prefix added to each one.

    """

    new_file_names = []

    for file_name in file_names:

        new_file_names.append(prefix + file_name)

    return new_file_names

