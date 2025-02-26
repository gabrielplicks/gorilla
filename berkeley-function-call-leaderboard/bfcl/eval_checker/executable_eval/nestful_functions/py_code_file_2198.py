import pickle

import hashlib

from typing import Any



def create_md5_hash(obj: Any, salt: str = "") -> str:

    """Calculates and returns the MD5 hash of a given object.

    The function uses `pickle` to serialize the object before passing it to the hash function.

    The function also supports an optional `salt` parameter to add a string to the serialized object before hashing it.

    If the `salt` parameter is not provided, an empty string is used.



    Args:

        obj: The object to calculate the MD5 hash for.

        salt: An optional string to add to the serialized object before hashing it.



    Returns:

        The MD5 hash of the serialized object.

    """

    serialized_obj = pickle.dumps(obj)

    salted_obj = serialized_obj + salt.encode('utf-8')

    return hashlib.md5(salted_obj).hexdigest()

