import base64

import pickle

import zlib



def serialize_base64_pickle_zlib(data: dict) -> str:

    """Serializes arbitrary data into a URL-safe string using base64, pickle, and zlib.



    Args:

        data: The data to be serialized.



    Returns:

        A URL-safe string containing the serialized data.

    """

    byte_string = pickle.dumps(data)

    compressed_string = zlib.compress(byte_string)

    base64_string = base64.urlsafe_b64encode(compressed_string).decode('utf-8')

    return base64_string

