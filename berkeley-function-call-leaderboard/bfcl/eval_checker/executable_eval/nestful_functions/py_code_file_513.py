import json

import hashlib



def hash_json(obj: dict) -> str:

    """Hashes a JSON-serializable object using the SHA-256 algorithm.



    Args:

        obj: The JSON-serializable object to hash.



    Returns:

        The hex-encoded hash digest of the input object.

    """

    json_str = json.dumps(obj)

    hash_obj = hashlib.sha256(json_str.encode('utf-8'))



    return hash_obj.hexdigest()

