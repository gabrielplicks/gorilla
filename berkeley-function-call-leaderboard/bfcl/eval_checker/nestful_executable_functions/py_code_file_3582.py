import os

import hashlib



def construct_hash_table(dir_path: str) -> dict:

    """Constructs a hash table consisting of file paths and their hash values.



    Args:

        dir_path: The directory path to traverse.



    Returns:

        A dictionary where the keys are the file paths and the values are the corresponding hash values.

    """

    hash_table = {}



    for root, dirs, files in os.walk(dir_path):

        for file_name in files:

            file_path = os.path.join(root, file_name)

            with open(file_path, 'rb') as f:

                content = f.read()



            hash_value = hashlib.md5(content).hexdigest()

            hash_table[file_path] = hash_value



    return hash_table

