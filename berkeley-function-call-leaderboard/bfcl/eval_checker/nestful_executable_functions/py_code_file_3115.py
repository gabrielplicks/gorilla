import os.path

from typing import Union



def is_video_file(filename: str) -> bool:

    """Checks if a file is a media file.



    Args:

        filename: The name of the file to check.



    Returns:

        True if the file extension is one of the common media file extensions,

        such as `.mp4`, `.avi`, `.mov`, `.mkv`, or `.mpg`, and False otherwise.

    """

    file_ext = os.path.splitext(filename)[1]

    if file_ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.mpg']:

        return True

    else:

        return False

