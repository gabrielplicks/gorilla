import hashlib

import binascii



def hex_sha256(data: str) -> str:

    """Computes the SHA-256 hash of a string and returns it as a hexadecimal string.



    Args:

        data: The input string.



    Returns:

        The SHA-256 hash of the input string as a hexadecimal string.

    """

    hashed = hashlib.sha256(data.encode()).digest()

    return binascii.hexlify(hashed).decode()

