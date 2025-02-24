import hashlib



def sha256_hash_file(file_bytes: bytes) -> str:

    """Calculates the SHA-256 hash of a file represented as a byte sequence.



    Args:

        file_bytes: The byte sequence representing the file.



    Returns:

        The hexadecimal string representation of the SHA-256 hash of the file.

    """

    chunk_size = 4096

    hasher = hashlib.sha256()

    for offset in range(0, len(file_bytes), chunk_size):

        chunk = file_bytes[offset:offset+chunk_size]

        hasher.update(chunk)



    return hasher.hexdigest().lower()

