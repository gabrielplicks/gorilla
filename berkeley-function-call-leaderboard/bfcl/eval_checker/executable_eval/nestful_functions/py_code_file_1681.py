import pickle



def load_dictionary(filename: str) -> dict:

    """Loads the contents of a pickled dictionary file into a Python dictionary.



    Args:

        filename: The name of the pickled dictionary file.



    Returns:

        The deserialized dictionary.

    """

    with open(filename, 'rb') as f:

        data = pickle.load(f)

    return data

