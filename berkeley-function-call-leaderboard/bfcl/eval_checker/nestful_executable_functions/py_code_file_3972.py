import pickle



class LoadDictionaryError(Exception):

    """Raised when there is an error loading a pickled dictionary."""

    pass



def load_dict(filename: str) -> dict:

    """Loads a pickled Python dictionary from a specified file and returns the dictionary.



    Args:

        filename: The name of the file to load the dictionary from.



    Raises:

        LoadDictionaryError: If there is an error opening or unpickling the file.

    """

    try:

        with open(filename, 'rb') as f:

            return pickle.load(f)

    except IOError as e:

        raise LoadDictionaryError(f'Error opening file: {e}')

    except pickle.UnpicklingError as e:

        raise LoadDictionaryError(f'Error unpickling data: {e}')

