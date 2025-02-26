import hashlib



def verify_password_hash(password: str, password_hash: str) -> bool:

    """Verifies if the provided password matches the password hash.



    Args:

        password: The user's password.

        password_hash: The password hash stored in the database.



    Returns:

        True if the password matches the hash, False otherwise.

    """

    hash_object = hashlib.sha256(password.encode())

    new_password_hash = hash_object.hexdigest()

    return new_password_hash == password_hash

