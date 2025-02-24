import hashlib



def get_sha256(file_path: str) -> str:

    """Calculates the SHA-256 hash of a file and outputs it as a string of hexadecimal digits.



    Args:

        file_path: The path to the file.



    Returns:

        The SHA-256 hash of the file as a string of hexadecimal digits.

    """

    hash_obj = hashlib.sha256()

    with open(file_path, 'rb') as f:

        while True:

            chunk = f.read(4096)

            if not chunk:

                break

            hash_obj.update(chunk)

    return hash_obj.hexdigest()

