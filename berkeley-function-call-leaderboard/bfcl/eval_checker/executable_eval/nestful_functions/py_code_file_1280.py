import hashlib



def calculate_sha256_checksum(file_path: str, block_size: int = 65536) -> str:

    """Calculates the checksum of a file using the SHA-256 algorithm.



    Args:

        file_path: The path to the file to be checked.

        block_size: The size of each block to read from the file. Default is 65536 bytes.



    Returns:

        The checksum of the file as a string in hexadecimal format.

    """

    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:

        while True:

            data = f.read(block_size)

            if not data:

                break

            sha256.update(data)

    return sha256.hexdigest()

