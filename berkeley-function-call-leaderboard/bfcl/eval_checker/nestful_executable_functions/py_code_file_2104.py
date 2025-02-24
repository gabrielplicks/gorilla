import os



def validate_csv_file_extension(filename: str) -> bool:

    """Checks if the filename extension is 'csv'.



    Args:

        filename: The name of the file to check.



    Returns:

        True if the filename extension is 'csv', False otherwise.

    """

    root, ext = os.path.splitext(filename)

    if ext == '.csv':

        return True

    else:

        return False

