import os



def join_file_paths(a: str, b: str) -> str:

    """Joins two file paths, a and b, into a single file path.



    Args:

        a: The first file path.

        b: The second file path.



    Returns:

        The joined file path.

    """

    a_normpath = os.path.normpath(a)

    b_normpath = os.path.normpath(b)

    if a_normpath.endswith(os.sep):

        a_normpath = a_normpath.rstrip(os.sep)

    if b_normpath.startswith(os.sep):

        b_normpath = b_normpath.lstrip(os.sep)

    joined_path = os.path.join(a_normpath, b_normpath)



    return joined_path

