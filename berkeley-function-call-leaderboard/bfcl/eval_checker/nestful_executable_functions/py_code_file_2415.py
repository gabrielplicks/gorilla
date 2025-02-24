import os

import hashlib



def hash_directory(directory: str) -> dict:

    """Computes the MD5 hashes of all files in a directory and its subdirectories.



    Args:

        directory: The path to the directory.



    Returns:

        A dictionary where the keys are the file paths and the values are their MD5 hashes.

    """

    file_hashes = {}



    for root, _, files in os.walk(directory):

        for filename in files:

            filepath = os.path.join(root, filename)

            with open(filepath, 'rb') as file:

                file_contents = file.read()

                file_hashes[filepath] = hashlib.md5(file_contents).hexdigest()



    return file_hashes

