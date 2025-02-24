import os



def same_file(path1: str, path2: str) -> bool:

    """Determines if two file paths resolve to the same file.



    Args:

        path1: The first file path.

        path2: The second file path.



    Returns:

        True if the two file paths resolve to the same file, False otherwise.

    """

    path1_real = os.path.realpath(path1)

    path2_real = os.path.realpath(path2)

    return path1_real == path2_real

