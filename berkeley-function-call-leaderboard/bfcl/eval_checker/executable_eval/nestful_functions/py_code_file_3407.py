import os



def delete_empty_folders(directory: str) -> None:

    """Deletes any empty subdirectories located within the specified directory.



    Args:

        directory: A string representing a directory path.

    """

    for root, dirs, files in os.walk(directory, topdown=False):

        for dir_name in dirs:

            dir_path = os.path.join(root, dir_name)

            if not os.listdir(dir_path):

                os.rmdir(dir_path)

