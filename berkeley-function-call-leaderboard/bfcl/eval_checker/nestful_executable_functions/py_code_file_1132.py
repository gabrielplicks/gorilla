import re



def change_extension_regex(filename: str, new_extension: str) -> str:

    """

    Changes the file extension of a given filename.



    Args:

        filename: The original filename.

        new_extension: The new file extension.



    Returns:

        The filename with the new extension.

    """

    extension_match = re.search(r"\.[^.]*$", filename)

    if extension_match:

        extension_length = len(extension_match.group())

        if extension_length >= len(new_extension):

            new_filename = re.sub(r"\.[^.]*$", "." + new_extension, filename)

            return new_filename

        else:

            new_filename = re.sub(r"\.[^.]*$", "." + new_extension, filename)

            return new_filename + new_extension[extension_length:]

    else:

        return filename

