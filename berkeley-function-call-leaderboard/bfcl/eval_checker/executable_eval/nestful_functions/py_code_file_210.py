import hashlib

import json

from typing import Any



def json_hash(data: Any) -> str:

    """Computes a unique hash for a given JSON string (a dictionary or list).



    Args:

        data: The input data to compute the hash for.



    Returns:

        A unique hash value for the input data.

    """

    serialized_data = json.dumps(data).encode("utf-8")

    hash_object = hashlib.md5(serialized_data)

    hash_value = hash_object.hexdigest()



    return hash_value

