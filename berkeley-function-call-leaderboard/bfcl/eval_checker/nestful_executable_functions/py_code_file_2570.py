import hashlib



def sha256sum(text: str) -> str:

    """Calculates the SHA256 hash value for a given text input.



    Args:

        text: The input text to calculate the SHA256 hash value for.



    Returns:

        The SHA256 hash value as a hexadecimal string.

    """

    hash_value = hashlib.sha256(text.encode()).hexdigest()

    return hash_value

