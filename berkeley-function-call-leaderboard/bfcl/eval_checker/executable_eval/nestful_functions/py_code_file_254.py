import hashlib



def generate_sha256_hash(string_to_hash: str) -> str:

    """Generates the SHA-256 hash of a string and encodes it as a hexadecimal string.



    Args:

        string_to_hash: The string to generate the hash of.



    Returns:

        The hexadecimal representation of the SHA-256 hash of the string.

    """

    hash = hashlib.sha256(string_to_hash.encode()).hexdigest()

    return hash

