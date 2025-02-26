import os



def walk_dir(dir_path: str) -> list:

    """Walks through a given directory and returns all the file paths as a list, excluding files starting with an underscore (_).



    Args:

        dir_path: The directory to walk through.



    Returns:

        A list of file paths.

    """

    file_paths = []



    for root, dirs, files in os.walk(dir_path):

        for file in files:

            if not file.startswith('_'):

                file_path = os.path.join(root, file)

                file_paths.append(file_path)



    return file_paths

