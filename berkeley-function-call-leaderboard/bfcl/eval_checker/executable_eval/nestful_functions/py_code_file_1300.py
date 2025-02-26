import os

import tempfile

from typing import Union



def temp_dir_path() -> Union[str, None]:

    """Creates a temporary directory in a system-dependent path and returns its absolute path.

    The function uses `tempfile.mkdtemp` to create the directory, and then concatenates its name

    with the directory's path using `os.path.join`. Finally, the function checks if the created

    directory exists, and if not, raises an exception with an appropriate message.

    Returns:

        The absolute path of the created temporary directory, or None if an exception is raised.

    """

    temp_dir = tempfile.mkdtemp()

    dir_path = os.path.join(tempfile.gettempdir(), os.path.basename(temp_dir))

    if not os.path.exists(dir_path):

        raise Exception(f"Directory {dir_path} does not exist")



    return dir_path

