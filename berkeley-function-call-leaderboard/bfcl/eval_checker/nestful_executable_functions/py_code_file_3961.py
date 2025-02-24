import os

from typing import List



def get_all_image_files(directory: str) -> List[str]:

    """

    Returns a list of all JPEG and PNG image files in the directory and its subdirectories.



    Args:

        directory: The directory path to search for image files.



    Returns:

        A list of full paths of all JPEG and PNG image files in the directory and its subdirectories.

    """

    image_extensions = ['.jpeg', '.png']

    image_files = []



    for root, dirs, files in os.walk(directory):

        image_files.extend([os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in image_extensions])



    return image_files

