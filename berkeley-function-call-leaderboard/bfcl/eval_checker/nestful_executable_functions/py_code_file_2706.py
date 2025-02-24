from typing import Union



def get_anchor_tag(file_path: str) -> str:

    """Generates an HTML anchor tag for a given source file path.



    Args:

        file_path: The path to the source file.



    Returns:

        An HTML anchor tag for the source file.



    Raises:

        Exception: If the file extension is not recognized.

    """

    file_extension = file_path.split(".")[-1]

    if file_extension == "py":

        link_path = file_path.replace(".py", ".html")

    elif file_extension == "html":

        link_path = file_path.replace(".html", ".html")

    else:

        raise Exception(f"Unrecognized file extension: {file_extension}")

    return f'<a href="{link_path}">{file_path.split("/")[-1]}</a>'

