from typing import Union

from hashlib import sha256



def secure_hash_password(password: str, channel_binding: bytes) -> Union[str, None]:

    """Computes the SHA256 hash of a password and a TLS channel binding.



    Args:

        password: The password to hash.

        channel_binding: The TLS channel binding.



    Returns:

        The resulting hash as a string, or a descriptive error message if an exception is raised.

    """

    try:

        hash_object = sha256(password.encode() + channel_binding)

        return hash_object.hexdigest()

    except Exception as e:

        return f"Error computing hash: {str(e)}"

