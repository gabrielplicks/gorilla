import pickle

import sys



def serialize_file(path: str) -> list:

    """Serializes the data of a file using pickle and returns a list containing the serialized data and its size.



    Args:

        path: The file path.



    Returns:

        A list containing the serialized data and its size.

    """

    try:

        with open(path, 'rb') as file:

            data = file.read()

            serialized_data = pickle.dumps(data)

            size = sys.getsizeof(serialized_data)

            return [serialized_data, size]

    except (FileNotFoundError, OSError):

        print("Invalid file path or error reading file.")

        return None

