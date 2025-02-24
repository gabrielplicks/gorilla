import hmac



def authenticate_token(token: str, secret_key: bytes, message: str) -> bool:

    """

    Authenticates a token by verifying its message and hmac.



    Args:

        token: A string of the format `{message}:{hmac}`.

        secret_key: The secret key used to generate the hmac.

        message: The message to be authenticated.



    Returns:

        True if the message and the hmac match, otherwise False.

    """

    message, hmac_token = token.rsplit(':', 1)

    computed_hmac = hmac.new(secret_key, message.encode(), digestmod='sha256').hexdigest()

    return hmac.compare_digest(computed_hmac, hmac_token)

