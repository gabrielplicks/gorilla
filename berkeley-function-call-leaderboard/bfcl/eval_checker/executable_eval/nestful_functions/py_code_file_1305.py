def file_type_callback(file_path: str) -> bool:

    """Determines whether a file is a CSV or TXT file based on its extension.



    Args:

        file_path: The path to the file.



    Returns:

        True if the file is a CSV or TXT file, False otherwise.

    """

    file_extension = file_path.split(".")[-1]

    if file_extension == "csv" or file_extension == "txt":

        return True

    return False

