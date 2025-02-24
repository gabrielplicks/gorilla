from typing import List

import pathlib



def filter_python_files(file_names: List[str]) -> List[str]:

    """

    Filters a list of file names and returns a new list containing only those file names that have the ".py" extension, regardless of case.

    Args:

        file_names: A list of file names.

    """

    return [file_name for file_name in file_names if pathlib.Path(file_name).suffix.lower() == '.py']

