import pickle

from typing import Any



def decode_code(code: str) -> str:

    """Decodes a 4-digit code into a string using a dictionary stored in a pickle file.



    Args:

        code: The 4-digit code to be decoded.



    Raises:

        ValueError: If the code cannot be decoded.

    """

    dictionary_filename = 'dictionary.pickle'

    with open(dictionary_filename, 'rb') as f:

        dictionary: dict[str, Any] = pickle.load(f)

    try:

        decoded_string = dictionary[code]

        return decoded_string

    except KeyError:

        raise ValueError(f'Code {code} cannot be decoded.')

