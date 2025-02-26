import hashlib

import base64

import hmac



def hmac_sha256(message: str, secret: str) -> str:

    """Generates a hash-based message authentication code (HMAC) using SHA256 for a given message, secret key, and base64 encoding.

    Args:

        message: The message to be authenticated.

        secret: The secret key used to generate the HMAC.

    """

    h = hmac.new(secret.encode(), message.encode(), digestmod=hashlib.sha256)

    digest = h.hexdigest()

    return base64.b64encode(digest.encode()).decode()

