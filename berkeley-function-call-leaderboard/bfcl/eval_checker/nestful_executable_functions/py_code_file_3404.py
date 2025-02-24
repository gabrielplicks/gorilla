import os

import typing



def get_secret_key() -> typing.Optional[str]:

    """Retrieves the secret key from an environment variable or a file.



    Args:

        None



    Returns:

        The secret key as a string, or None if the secret key could not be found.

    """

    secret_key = os.getenv("SECRET_KEY")



    if not secret_key:

        with open('/secrets/secret_key.txt') as f:

            secret_key = f.read().strip()



    return secret_key

