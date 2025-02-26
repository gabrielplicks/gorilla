import hashlib



def calculate_hash_value(string: str, hash_func: str) -> int:

    """Calculates the hash value of a string using the specified hash function.

    Args:

        string: The string to calculate the hash value for.

        hash_func: The hash function to use. Supported hash functions are: "sha256", "sha512", "blake2b", "blake2s", and "sha3_256".

    """

    try:

        if hash_func not in ["sha256", "sha512", "blake2b", "blake2s", "sha3_256"]:

            raise ValueError("Invalid hash function")

        hash_value = int(eval(f"hashlib.{hash_func}(string.encode()).hexdigest()"), 16)

        return hash_value

    except Exception as e:

        raise e

