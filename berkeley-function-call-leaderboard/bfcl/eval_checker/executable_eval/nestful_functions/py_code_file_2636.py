from typing import AnyStr



def consistent_hash(key: AnyStr, max_hash_code: int) -> int:

    """Calculates a consistent hash code for a given key.

    Args:

        key: The key to be hashed.

        max_hash_code: The maximum possible hash code.

    """

    hash_code = 0

    for char in key:

        hash_code = (hash_code * 31 + ord(char)) % max_hash_code

    return hash_code

