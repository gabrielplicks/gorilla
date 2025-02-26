from typing import List



def append_extension(file_names: List[str], ext: str) -> List[str]:

    """Appends the extension `ext` to each of the file names and returns the new list.



    Args:

        file_names: A list of file names.

        ext: The extension to append to each file name.

    """

    new_file_names = []



    for file_name in file_names:

        new_file_name = file_name + ext

        new_file_names.append(new_file_name)



    return new_file_names

