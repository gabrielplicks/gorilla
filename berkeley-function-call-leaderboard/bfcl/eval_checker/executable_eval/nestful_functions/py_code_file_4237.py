import zipfile

from typing import Union



def count_files_in_zip(zip_path: str) -> Union[int, str]:

    """Counts the number of files in a ZIP file.

    Args:

        zip_path: The path to the ZIP file.

    Returns:

        The number of files in the ZIP file, or an error message if the file is not found or is not a valid ZIP file.

    """

    try:

        with zipfile.ZipFile(zip_path, mode='r') as zip_file:

            return len(zip_file.namelist())

    except FileNotFoundError:

        return "No such file or directory"

    except zipfile.BadZipFile:

        return "Not a valid ZIP file"

