import os



def join_file_name(file_name: str, file_path: str) -> str:

    """Joins the file_name and file_path using a file system's path separator.

    Args:

        file_name: The name of the file.

        file_path: The path to the file.

    """

    if file_path is None or file_path == "":

        return file_name

    return os.path.join(file_path, file_name)

