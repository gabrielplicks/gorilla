from typing import Dict, Any



def is_hash(string: str) -> bool:

    """Determines if a given string represents a hash of a supported type.



    Args:

        string: The string to check.



    Returns:

        True if the string represents a hash of a supported type, False otherwise.

    """

    supported_hashes: Dict[str, int] = {

        "sha256": 64,

        "sha512": 128,

    }

    string_length: int = len(string)

    if string_length not in supported_hashes.values():

        return False

    for hash_type, length in supported_hashes.items():

        if string_length == length and all(char in "0123456789abcdef" for char in string):

            return True



    return False

