import base64

import secrets

import typing



characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"



def generate_and_encode() -> typing.Tuple[bytes, int]:

    """Generates a random sequence of 12 characters from the base64 alphabet and encodes it using base64.

    Returns the encoded string as a Python byte string.



    Args:

        None



    Returns:

        A tuple containing the encoded byte string and the length of the encoded byte string.

    """

    random_sequence = "".join(secrets.choice(characters) for _ in range(12))

    encoded_sequence = base64.b64encode(random_sequence.encode())

    return encoded_sequence, len(encoded_sequence)

