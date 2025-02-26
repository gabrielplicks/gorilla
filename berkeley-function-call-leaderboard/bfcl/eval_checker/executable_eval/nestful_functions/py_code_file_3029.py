import os

import glob



def delete_temp_files(directory: str) -> None:

    """Deletes all files with .tmp extensions in the given directory.



    Args:

        directory: The directory to search for temporary files.



    Raises:

        FileNotFoundError: If the directory does not exist.

        OSError: If a file cannot be deleted.

    """

    if not os.path.exists(directory):

        raise FileNotFoundError("Directory does not exist")

    temp_files = glob.glob(os.path.join(directory, "*.tmp"))

    for file in temp_files:

        try:

            os.remove(file)

        except OSError as e:

            raise e

