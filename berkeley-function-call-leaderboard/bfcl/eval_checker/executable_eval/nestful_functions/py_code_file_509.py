import hashlib

import json



def hash_metadata(file_path: str, metadata: dict) -> str:

    """Computes the SHA256 hash of a file at a given path by concatenating its contents with the string representation of the provided metadata dictionary.

    The function encodes the file contents and metadata as UTF-8 strings before hashing.



    Args:

        file_path: The path to the file.

        metadata: The metadata dictionary.



    Returns:

        The hexadecimal representation of the SHA256 hash.

    """

    with open(file_path, 'rb') as file:

        file_contents = file.read()



    encoded_metadata = json.dumps(metadata).encode('utf-8')

    data_to_hash = file_contents + encoded_metadata



    sha256 = hashlib.sha256()

    sha256.update(data_to_hash)

    hex_hash = sha256.hexdigest()



    return hex_hash

