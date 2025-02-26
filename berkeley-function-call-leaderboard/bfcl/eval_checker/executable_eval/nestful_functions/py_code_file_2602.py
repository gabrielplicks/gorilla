import os

import hashlib



def md5_and_size(filename: str) -> dict:

    """Calculates the MD5 hash and size of a file.



    Args:

        filename: The name of the file to calculate the hash and size for.



    Returns:

        A dictionary containing the MD5 hash and size of the file.

    """

    with open(filename, 'rb') as f:

        file_size = os.path.getsize(filename)

        contents = f.read()

        hash_contents = hashlib.md5(contents).hexdigest()

    return {

        'md5': hash_contents,

        'size': file_size,

    }

