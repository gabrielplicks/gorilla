import hashlib

from typing import Tuple



def compare_file_hashes(file1: str, file2: str) -> bool:

    """Compares the content of two files using their content's hash values.



    Args:

        file1: The path to the first file.

        file2: The path to the second file.



    Returns:

        True if the hash values are the same, indicating the contents of the files are equal.

        False otherwise.

    """

    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:

        try:

            hash1 = hashlib.sha256(f1.read()).hexdigest()

            hash2 = hashlib.sha256(f2.read()).hexdigest()

            return hash1 == hash2

        except Exception as e:

            print(f"Error: {e}")

            return False

