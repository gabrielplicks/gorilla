import pickle

import os



def load_pickle_file(file_path: str) -> object:

    """Loads a pickle file into a Python object.



    Args:

        file_path: The path to the pickle file.



    Returns:

        The object of the file's contents if the file exists and can be unpickled,

        otherwise None.

    """

    if not os.path.exists(file_path):

        print(f"Error: File does not exist: {file_path}")

        return None



    try:

        with open(file_path, "rb") as file:

            return pickle.load(file)

    except (FileNotFoundError, pickle.UnpicklingError) as e:

        print(f"Error: Unable to load file: {e}")

        return None

