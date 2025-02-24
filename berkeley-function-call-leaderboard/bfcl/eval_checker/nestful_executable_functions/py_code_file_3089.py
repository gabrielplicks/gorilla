import os

from typing import Tuple



IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp',

                    '.svg', '.tiff', '.tif', '.ico')



def keep_only_images(path: str) -> None:

    """Removes all non-image files from the given directory and its subdirectories.



    Args:

        path: The directory path.

    """

    for root, dirs, files in os.walk(path):

        for file in files:

            file_path = os.path.join(root, file)

            ext: Tuple[str, str] = os.path.splitext(file_path)

            if ext[1].lower() not in IMAGE_EXTENSIONS:

                os.remove(file_path)

