import os

import shutil

import tempfile



def copy_and_return(src_file: str, dst_dir: str) -> tuple:

    """Creates a temporary directory, copies a file from the local file system into it, and returns both the path to the temporary directory and the path to the copied file.



    Args:

        src_file: The path to the source file on the local file system.

        dst_dir: The destination directory path.



    Returns:

        A tuple containing the temporary directory path and the copied file path.

    """

    temp_dir = tempfile.mkdtemp()

    shutil.copy(src_file, temp_dir)

    dst_file = os.path.join(temp_dir, os.path.basename(src_file))

    return temp_dir, dst_file

