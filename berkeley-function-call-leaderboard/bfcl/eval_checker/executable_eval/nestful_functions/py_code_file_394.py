import os



def get_path_info(path: str) -> dict:

    """

    Returns a dictionary containing the directory name, file name, and file extension from a given file path.

    Args:

        path: The file path.

    """

    filename, extension = os.path.splitext(path)

    directory, filename = os.path.split(filename)



    info = {"directory": directory, "filename": filename}

    if extension:

        info["extension"] = extension[1:]  # Remove the dot from the extension



    return info

