import os

import re



def clean_up_path(path: str) -> str:

    """

    Removes redundant dots from a file path.



    Args:

        path: The file path to clean up.



    Returns:

        The cleaned up file path.

    """

    directory, file = os.path.split(path)

    new_path = os.path.join(directory, file)

    new_path = re.sub(r"\.", "", new_path)



    return new_path

