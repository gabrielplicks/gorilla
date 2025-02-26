import os

from typing import List



def create_directory_with_subdirectories(directory_path: str, subdirectories: List[str]) -> bool:

    """

    Creates the specified directory and all of its subdirectories if they do not exist.

    Returns True if the directory and all subdirectories were successfully created, and

    False if any errors occurred.

    Args:

        directory_path: The path of the directory to be created.

        subdirectories: A list of subdirectories to be created within the directory.

    """

    try:

        os.makedirs(directory_path, exist_ok=True)

        for subdir in subdirectories:

            os.makedirs(os.path.join(directory_path, subdir), exist_ok=True)

        return True

    except Exception as e:

        print(f"Error creating directory: {e}")

        return False

