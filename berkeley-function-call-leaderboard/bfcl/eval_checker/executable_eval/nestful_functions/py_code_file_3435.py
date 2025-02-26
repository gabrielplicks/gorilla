import os

import zipfile



def check_zip_and_unzip(path: str) -> bool:

    """Checks if a given path contains a zip file and if it does, unzips the contents to a new location.



    Args:

        path: The path to check for a zip file.



    Returns:

        A boolean indicating whether the zip file was successfully unzipped.

    """

    if os.path.isfile(path) and zipfile.is_zipfile(path):

        new_directory = os.path.splitext(path)[0] + "_unzipped"  # Use the same name as the zip file with "_unzipped" appended

        with zipfile.ZipFile(path) as zip_file:

            zip_file.extractall(new_directory)

        return True

    else:

        return False

