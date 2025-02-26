import pickle



def read_pickled_file(filename: str) -> dict:

    """Reads a pickled file and returns a dictionary containing the data.



    Args:

        filename: The name of the pickled file.



    Raises:

        ValueError: If the file does not contain a dictionary.

    """

    with open(filename, 'rb') as f:

        data = pickle.load(f)



    if not isinstance(data, dict):

        raise ValueError("The file does not contain a dictionary.")



    return data

