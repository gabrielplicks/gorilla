import hashlib



def string_to_md5(string: str) -> str:

    """Converts a given string into its MD5 hash and returns the result as a lowercase hex string.



    Args:

        string: The input string to be converted into its MD5 hash.



    Returns:

        The MD5 hash of the input string as a lowercase hex string.

    """

    hash_obj = hashlib.md5(string.encode('utf-8'))

    md5_hash = hash_obj.hexdigest()

    return md5_hash

