import hashlib



def generate_sha256(filepath: str) -> str:

    """Generates a SHA256 checksum from a file path.



    Args:

        filepath: The path to the file.



    Returns:

        The SHA256 checksum as a string.

    """

    with open(filepath, "rb") as f:

        contents = f.read()

        hash_obj = hashlib.sha256(contents)

        return hash_obj.hexdigest()

