import hashlib



def md5_digest(filename: str) -> str:

    """Returns the MD5 digest of a given file.



    Args:

        filename: The name of the file to compute the MD5 digest for.

    """

    with open(filename, 'rb') as f:

        data = f.read()

    md5 = hashlib.md5(data)

    return md5.hexdigest()

