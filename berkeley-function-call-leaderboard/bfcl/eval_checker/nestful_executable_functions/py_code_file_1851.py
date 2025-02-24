import os



def delete_empty_directories(root_dir: str) -> None:

    """

    Traverses a directory structure and deletes all empty directories.



    Args:

        root_dir: The root directory path.

    """

    try:

        for dir_path, dir_list, file_list in os.walk(root_dir):

            if len(dir_list) == 0 and len(file_list) == 0:

                try:

                    os.rmdir(dir_path)

                except OSError:

                    print(f"Warning: Directory {dir_path} could not be deleted.")

    except OSError:

        print(f"Warning: Directory traversal could not be completed.")

