import hashlib

from typing import Tuple



def are_files_identical(file1: str, file2: str) -> bool:

    """Determines if two files are identical by comparing their SHA-256 hashes.



    Args:

        file1: The path to the first file.

        file2: The path to the second file.



    Returns:

        True if the files are identical, False otherwise.

    """

    try:

        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:

            hash1 = hashlib.sha256(f1.read()).hexdigest()

            hash2 = hashlib.sha256(f2.read()).hexdigest()

            return hash1 == hash2

    except FileNotFoundError as e:

        print(f'File not found: {e}')

    except PermissionError as e:

        print(f'Permission denied: {e}')

    return False

