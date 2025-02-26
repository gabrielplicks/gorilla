import pickle

from typing import Tuple



def load_pickle_and_get_string_bytes(pickle_file: str) -> Tuple[str, int]:

    """Loads data from a pickle file and returns a tuple containing two elements.

    The first element is the string representation of the loaded data,

    and the second element is the number of bytes in that string.

    Args:

        pickle_file: The path to the pickle file.

    """

    with open(pickle_file, 'rb') as f:

        data = pickle.load(f)

        string_representation = str(data)

        num_bytes = len(string_representation)

        return (string_representation, num_bytes)

