from typing import Dict, List



def extract_file_extensions(file_paths: List[str]) -> Dict[str, int]:

    """Extracts file extensions from a list of file paths.



    Args:

        file_paths: A list of file paths.



    Returns:

        A dictionary where the keys are the file extensions and the values are the number of occurrences of each extension.

    """

    extensions = {}

    for path in file_paths:

        extension = path.split('.')[-1]

        extensions[extension] = extensions.get(extension, 0) + 1

    return extensions

