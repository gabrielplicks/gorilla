from typing import List



def find_txt_files(file_paths: List[str]) -> List[str]:

    """Finds the first file path with a `.txt` extension in a list of file paths.



    Args:

        file_paths: A list of file paths.



    Returns:

        A list of file paths with a `.txt` extension.

    """

    return [path for path in file_paths if path.endswith(".txt")]

