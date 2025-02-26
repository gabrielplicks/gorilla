import re



def is_pdf_file(filename: str) -> tuple[bool, str]:

    """Checks whether a given filename is a PDF file or not.

    Args:

        filename: The name of the file.

    Returns:

        A tuple containing a boolean indicating whether the file is a PDF and the name of the file without the extension.

    """

    match = re.match(r"^(.*)\.pdf$", filename)

    if match:

        return True, match.group(1)

    else:

        return False, filename

