import hashlib



def create_hash_md5(string: str) -> str:

    """Creates a hash of a given string based on the MD5 algorithm.



    Args:

        string: The input string to be hashed.



    Returns:

        The MD5 hash of the input string as a hexadecimal string.



    Raises:

        Exception: If any error occurs during encoding or hashing.

    """

    try:

        encoded_string = string.encode("utf-8")

        hash_md5 = hashlib.md5(encoded_string).hexdigest()



        return hash_md5

    except Exception as e:

        print(f"Error: {e}")

        return "Error occurred while hashing"

