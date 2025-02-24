from typing import List



def count_images(paths: str, extension: str) -> int:

    """Counts the number of image files with a specific extension in a string of paths.



    Args:

        paths: A string containing multiple paths to image files separated by a delimiter.

        extension: The extension of the image files to count.

    """

    paths = paths.replace(" ", "")

    normalized_paths: List[str] = [path.strip().lower() for path in paths.split(",")]

    count: int = 0

    for path in normalized_paths:

        if path.endswith(extension):

            count += 1



    return count

