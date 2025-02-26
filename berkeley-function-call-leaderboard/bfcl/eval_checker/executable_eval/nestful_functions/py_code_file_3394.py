import pickle

from typing import Dict



def load_uuid_to_user_id_map() -> Dict[str, str]:

    """Loads a dictionary from a pickle file that maps UUIDs to user IDs.

    If the file is not found, an empty dictionary is returned.

    Returns:

        A dictionary mapping UUIDs to user IDs.

    """

    try:

        with open('uuid_to_user_id_map.pickle', 'rb') as pickle_file:

            return pickle.load(pickle_file)

    except FileNotFoundError:

        return {}

