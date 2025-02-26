from typing import Union



def get_title_from_filename(filename: str) -> Union[str, None]:

    """Extracts the title from a file name.



    The file name is in the format `<title>.<extension>`. The function returns the title.



    Args:

        filename: The file name.



    Returns:

        The title of the file. If the file name does not have an extension, returns None.

    """

    parts = filename.split('.')

    if len(parts) > 1:

        return parts[0]

    else:

        return None

