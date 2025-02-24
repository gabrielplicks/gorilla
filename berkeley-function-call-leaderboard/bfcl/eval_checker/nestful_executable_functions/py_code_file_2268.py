import hashlib



def file_hash(filepath: str) -> str:

    """

    Returns the MD5 hash of the file contents at `filepath` in hexadecimal format.



    Args:

        filepath: The path to the file.

    """

    BLOCK_SIZE = 65536  # Adjust as needed

    hasher = hashlib.md5()



    with open(filepath, 'rb') as fp:

        while True:

            data = fp.read(BLOCK_SIZE)

            if not data:

                break

            hasher.update(data)



    return hasher.hexdigest()

