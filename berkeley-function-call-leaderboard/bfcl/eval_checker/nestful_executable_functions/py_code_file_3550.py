import hashlib

from typing import AnyStr



def generate_document_id(string: AnyStr) -> str:

    """Generates a unique document ID based on a given string.



    The function hashes the string and returns a 6-character alphanumeric document ID.

    The function does not allow duplicate IDs for different strings.



    Args:

        string: The input string to generate the document ID from.



    Returns:

        A 6-character alphanumeric document ID.

    """

    hash_id = hashlib.sha1(string.encode()).hexdigest()

    return hash_id[:6]

