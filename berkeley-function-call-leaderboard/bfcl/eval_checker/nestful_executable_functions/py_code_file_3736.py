import re



def file_extension_match(file_name: str, extension_list: list[str]) -> bool:

    """Checks if a file name ends with any of the valid extensions.



    Args:

        file_name: The name of the file.

        extension_list: A list of valid extensions.

    """

    for extension in extension_list:

        if re.search(fr'\.{extension}$', file_name):

            return True

    return False

