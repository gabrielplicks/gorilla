import hashlib



def calculate_file_checksum(fname: str) -> str:

    """Calculates the checksum (SHA-256) of a given file.



    Args:

        fname: The name of the file to calculate the checksum for.



    Returns:

        The checksum of the file as a string.

    """

    with open(fname, "rb") as f:

        hasher = hashlib.sha256()

        chunk_size = 65536  # 64 KB

        chunk = f.read(chunk_size)

        while chunk:

            hasher.update(chunk)

            chunk = f.read(chunk_size)

        checksum = hasher.hexdigest()

    return checksum

