import hashlib



def compute_sha256_hex(input_string: str) -> str:

    """Computes the SHA-256 hash of a given input string using the hashlib library.

    Returns the resulting string as a hexadecimal representation of the binary data.

    Args:

        input_string: The input string to compute the SHA-256 hash.

    """

    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()

    return sha256_hash

