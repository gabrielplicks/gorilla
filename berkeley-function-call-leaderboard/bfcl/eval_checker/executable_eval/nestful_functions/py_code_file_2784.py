import hashlib



def compute_md5(string: str) -> str:

    """Computes the MD5 digest of a given string.



    Args:

        string: The input string.



    Returns:

        A 32-character hex-encoded message digest.

    """

    hash_obj = hashlib.md5()

    hash_obj.update(string.encode('utf-8'))

    return hash_obj.hexdigest()

