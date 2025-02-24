import os.path



def replace_extension(file_path: str, new_extension: str) -> str:

    """Replaces the extension of a file path with a new extension.



    Args:

        file_path: The original file path.

        new_extension: The new extension to replace the original extension.

    """

    file_name = os.path.basename(file_path)

    file_name, old_extension = os.path.splitext(file_name)

    directory_path = os.path.dirname(file_path)

    new_file_name = f"{file_name}{new_extension}"

    new_path = os.path.join(directory_path, new_file_name)

    return new_path

