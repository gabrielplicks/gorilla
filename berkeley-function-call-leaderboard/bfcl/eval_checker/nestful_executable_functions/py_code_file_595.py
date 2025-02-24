import json

import hashlib



def generate_deterministic_hash(json_dict: dict) -> str:

    """Generates a deterministic hash from a JSON dictionary.



    Args:

        json_dict: The JSON dictionary to generate a hash from.



    Returns:

        The hexadecimal string representation of the hash.

    """

    serialized_dict = json.dumps(json_dict, sort_keys=True)

    hash_obj = hashlib.sha256(serialized_dict.encode('utf-8'))

    return hash_obj.hexdigest()

