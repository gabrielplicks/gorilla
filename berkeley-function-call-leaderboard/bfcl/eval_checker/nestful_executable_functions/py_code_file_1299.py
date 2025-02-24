from typing import List



def hash_string(string: str) -> int:

    """Calculates the hash value of a string by summing the ordinal values of its characters.



    Args:

        string: The input string.



    Returns:

        The hash value of the string.

    """

    return sum(ord(char) for char in string)



def encode_string(hashed_string: List[int]) -> str:

    """Decodes a list of hash values back into the original string.



    Args:

        hashed_string: A list of hash values.



    Returns:

        The original string.

    """

    return ''.join(chr(hash) for hash in hashed_string)

