import os

from typing import List



def is_image(filename: str) -> bool:

    """Checks if the given filename is an image file.



    Args:

        filename: The name of the file to check.



    Returns:

        True if the file is an image, False otherwise.

    """

    file_extension = os.path.splitext(filename)[1].lower()

    supported_image_formats: List[str] = [".jpg", ".png", ".gif", ".tif", ".bmp"]

    return file_extension in supported_image_formats

