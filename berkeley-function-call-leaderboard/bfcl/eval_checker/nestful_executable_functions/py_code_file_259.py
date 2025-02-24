import hashlib



def get_file_sha1(file_path: str) -> str:

    """Calculates the sha1 hash of a file's contents.



    Args:

        file_path: The path to the file.



    Returns:

        The sha1 hash of the file's contents.

    """

    with open(file_path, 'rb') as f:

        sha1 = hashlib.sha1()

        while True:

            chunk = f.read(4096)  # Read the file in chunks

            if not chunk:

                break

            sha1.update(chunk)

        return sha1.hexdigest()

